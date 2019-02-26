function eval_thumos14()
%

% ----------------------------------------------------------------------------------------------------------------
% Segment-CNN
% Copyright (c) 2016 , Digital Video & Multimedia (DVMM) Laboratory at Columbia University in the City of New York.
% Licensed under The MIT License [see LICENSE for details]
% Written by Zheng Shou, Dongang Wang, and Shih-Fu Chang.
% ----------------------------------------------------------------------------------------------------------------

clear;

AP_all=cell(0);
PR_all=cell(0);
mAP=[];
REC_all=[];

detfilename = '../tmp.txt';
gtpath = './annotation/annotation_test/';
subset = 'test';
new_length = 16;
test_thresholds =   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
nms_thresholds =    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
% nms_thresholds =    [0.02, 0.06, 0.08, 0.12, 0.3, 0.4, 0.4];
cls_thresh = 0.000001;
pro_thresh = 0.1;
video_num = 213;
proposals_per_video = 200;
use_reg = false;
for index = 1:length(test_thresholds)
    threshold = test_thresholds(index);
    fprintf('\n threshold=%f \n', threshold)    
    [videonames,t1,t2,clsid,conf]=textread(detfilename,'%s%f%f%d%f');
    %step1: For multilabel: assign all CliffDiving label for Diving label
    %cliffDiving = clsid==5;
    %videonames = [videonames; videonames(cliffDiving)];
    %t1 = [t1; t1(cliffDiving)];
    %t2 = [t2; t2(cliffDiving)];
    %clsid = [clsid; 8*ones(sum(cliffDiving),1)];
    %conf = [conf; conf(cliffDiving)];
    %step2: Fillter confidence lower than the threshold
    confid = conf>0.005; % visualize thresh0.6
    videonames = videonames(confid);
    t1 = t1(confid);
    t2 = t2(confid);
    clsid = clsid(confid);
    conf = conf(confid);
    %step3: NMS after detection -per video
    overlap_nms = nms_thresholds(index);
	videoid = unique(videonames);
	tic; 
	pick_nms = [];
	for id=1:length(videoid)
        vid = videoid{id};
		for cls=1:20
			inputpick = find((strcmp(videonames,vid))&(clsid==cls));
			pick_nms = [pick_nms; inputpick(nms_temporal([t1(inputpick) ...
				,t2(inputpick),conf(inputpick)],overlap_nms))]; 
            %pick_nms = [pick_nms; inputpick(soft_nms_temporal([t1(inputpick) ...
		    % ,t2(inputpick),conf(inputpick)],'Nt',overlap_nms,'method',1))]; 
		end
	end
	toc;
    pick_nms = pick_nms(1:min(length(pick_nms), proposals_per_video*video_num));
    videonames = videonames(pick_nms);
    t1 = t1(pick_nms);
    t2 = t2(pick_nms);
    clsid = clsid(pick_nms); 
    conf = conf(pick_nms);
	% ===============================

% 	% rank score by overlap score
% 	seg_swin_test = seg_swin_test([pick_nms],:);
% 	[~,order]=sort(-seg_swin_test(:,9));
% 	seg_swin_test = seg_swin_test(order,:);
    
    fout = fopen('tmp_run.txt', 'w');
    for i=1:length(videonames)
        fprintf(fout, [videonames{i} ' ' num2str(t1(i)) ' ' num2str(t2(i)) ' ' num2str(clsid(i)) ' ' num2str(conf(i)) '\n']);
    end
    fclose(fout);
    [pr_all,ap_all,map] = TH14evalDet_Updated('tmp_run.txt',gtpath,subset,threshold);
    mAP=[mAP,map];
    AP_all{end+1}=ap_all;
    PR_all{end+1}=pr_all;
    ave_rec = 0;
    for ii=1:20
        ave_rec = ave_rec + pr_all(ii).rec(end);
    end
    ave_rec = ave_rec/20;
    REC_all=[REC_all,ave_rec];
end

save('res_thumos14.mat','PR_all','AP_all','REC_all','mAP');

