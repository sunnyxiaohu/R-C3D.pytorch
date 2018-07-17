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
thresholds = 0.5:0.05:0.95;
detfilename = '../tmp.txt';
gtpath = './annotation/annotation_test/';
subset = 'test';
video_num = 213;
proposals_per_video = 100;
use_tag = false;
use_reg = false;
for threshold = thresholds
    [videonames,t1,t2,clsid,conf]=textread(detfilename,'%s%f%f%d%f');
    %step2: Fillter confidence lower than the threshold
    confid = conf>0.005;
    videonames = videonames(confid);
    t1 = t1(confid);
    t2 = t2(confid);
    clsid = clsid(confid);
    conf = conf(confid);
    %step3: NMS after detection -per video
    overlap_nms = 0.6;
	videoid = unique(videonames);
	tic; 
	pick_nms = [];
	for id=1:length(videoid)
        vid = videoid{id};
		for cls=100 %1:20
        %for cls=1:20
			inputpick = find((strcmp(videonames,vid))&(clsid==cls));
			pick_nms = [pick_nms; inputpick(nms_temporal([t1(inputpick) ...
				,t2(inputpick),conf(inputpick)],overlap_nms))]; 
		end
	end
	toc;
    videonames = videonames(pick_nms);
    t1 = t1(pick_nms);
    t2 = t2(pick_nms);
    clsid = clsid(pick_nms); 
    conf = conf(pick_nms);
    
    % add context for recall evaluation
    t1_ctx06 = max(0, t1-(0.6-1)*(t2-t1)*0.5);
    t2_ctx06 = max(0, t2+(0.6-1)*(t2-t1)*0.5);
    t1_ctx08 = max(0, t1-(0.8-1)*(t2-t1)*0.5);
    t2_ctx08 = max(0, t2+(0.8-1)*(t2-t1)*0.5);
    t1_ctx15 = max(0, t1-(1.5-1)*(t2-t1)*0.5);
    t2_ctx15 = max(0, t2+(1.5-1)*(t2-t1)*0.5);
    t1_ctx21 = max(0, t1-(2.1-1)*(t2-t1)*0.5);
    t2_ctx21 = max(0, t2+(2.1-1)*(t2-t1)*0.5);
    t1 =  [t1_ctx06]; %[t1; t1_ctx08; t1_ctx15; t1_ctx21];
    t2 = [t2_ctx06]; %[t2; t2_ctx08; t2_ctx15; t2_ctx21]; 
    clsid = [clsid]; % clsid; clsid; clsid];
    conf = [conf]; % conf; conf; conf];
    videonames = [videonames]; %videonames; videonames; videonames];
    
    % select topN proposals per video.
    videonames = videonames(1:min(video_num*proposals_per_video,length(videonames)));
    t1 = t1(1:min(video_num*proposals_per_video,length(videonames)));
    t2 = t2(1:min(video_num*proposals_per_video,length(videonames)));
    clsid = clsid(1:min(video_num*proposals_per_video,length(videonames))); 
    conf = conf(1:min(video_num*proposals_per_video,length(videonames)));
	% ===============================
    
    fout = fopen('tmp_run.txt', 'w');
    for i=1:length(videonames)
        % class_id=100, class='Agnostic' for proposal evaluation.
        fprintf(fout, [videonames{i} ' ' num2str(t1(i)) ' ' num2str(t2(i)) ' 100 ' num2str(conf(i)) '\n']);
    end
    fclose(fout);

    [rec, prec] = TH14evalProp('tmp_run.txt',gtpath,subset,threshold);
    REC_all = [REC_all, rec(end)];
end
% mean(REC_all(5:10))
mean(REC_all(:))
figure(1);

% set(gca,'YLabel','FontSize',8,'Vertical','middle');
plot(thresholds, REC_all, 'LineWidth', 3);
title(['AR[0.5:0.1:1]=' num2str(mean(REC_all(:)))], 'FontWeight', 'bold')
xlabel('tIoU', 'FontSize',12, 'FontWeight','bold');
ylabel('recall', 'FontSize',12, 'FontWeight','bold');
%set(gca, 'XTickLabel',[0,0.7,1], 'YTickLabel',[0.2,0.4,0.6,0.8]);%坐标标签
set(gca,'FontSize',12);
set(gca, 'XTick',[0:0.2:1], 'YTick',[0:0.2:1]);%要显示的坐标刻度
