function [pr_all,ap_all,map]=TH14evalDet_Updated(detfilename,gtpath,subset,threshold)
% [pr_all,ap_all,map]=TH14evalDet_Updated(detfilename,gtpath,subset,[threshold])
%
%     Input:    detfilename: file path of the input file
%                    gtpath: the path of the groundtruth directory
%                    subset: 'test' or 'val', means the testset or validation set
%                 threshold: overlap threshold (0.5 in default)
%
%    Output:         pr_all: precision-recall curves
%                    ap_all: AP for each class
%                       map: MAP 
%
%
% Evaluation of the temporal detection for 20 classes in the THUMOS 2014 
% action detection challenge http://crcv.ucf.edu/THUMOS14/
%
% The function produces precision-recall curves and average precision
% values for each action class and five values of thresholds for
% the overlap  between ground-truth action intervals and detected action
% intervals. Mean average precision values over classes are also returned.
% 
%
%
% Example:
%
%  [pr_all,ap_all,map]=TH14evalDet('results/Run-2-det.txt','annotation','test',0.5);
%
%
% Plotting precision-recall results:
% 
%  overlapthresh=0.1;
%  ind=find([pr_all.overlapthresh]==overlapthresh);
%  clf
%  for i=1:length(ind)
%    subplot(4,5,i)
%    pr=pr_all(ind(i));
%    plot(pr.rec,pr.prec)
%    axis([0 1 0 1])
%    title(sprintf('%s AP:%1.3f',pr.class,pr.ap))
%  end 
%
  
% THUMOS14 detection classes
%
  
if nargin<4
    threshold=0.5;
end
if nargin<3
    error('At least 3 parameters!')
end


[th14classids,th14classnames]=textread([gtpath '/detclasslist.txt'],'%d%s');
  
% read ground truth
%

clear gtevents
gteventscount=0;
th14classnamesamb=cat(1,th14classnames,'Ambiguous');
for i=1:length(th14classnamesamb)
  class=th14classnamesamb{i};
  gtfilename=[gtpath '/' class '_' subset '.txt'];
  if exist(gtfilename,'file')~=2
    error(['TH14evaldet: Could not find GT file ' gtfilename])
  end
  [videonames,t1,t2]=textread(gtfilename,'%s%f%f');
  for j=1:length(videonames)
    gteventscount=gteventscount+1;
    gtevents(gteventscount).videoname=videonames{j};
    gtevents(gteventscount).timeinterval=[t1(j) t2(j)];
    gtevents(gteventscount).class=class;
    gtevents(gteventscount).conf=1;
  end
end


% parse detection results
%

if exist(detfilename,'file')~=2
  error(['TH14evaldet: Could not find file ' detfilename])
end

[videonames,t1,t2,clsid,conf]=textread(detfilename,'%s%f%f%d%f');
videonames=regexprep(videonames,'\.mp4','');

clear detevents
for i=1:length(videonames)
  ind=find(clsid(i)==th14classids);
  if length(ind)
    detevents(i).videoname=videonames{i};
    detevents(i).timeinterval=[t1(i) t2(i)];
    detevents(i).class=th14classnames{ind};
    detevents(i).conf=conf(i);
  else
    fprintf('WARNING: Reported class ID %d is not among THUMOS14 detection classes.\n')
  end
end

% Evaluate per-class PR for multiple overlap thresholds
%

ap_all=[];
clear pr_all

for i=1:length(th14classnames)
  class=th14classnames{i};
  classid=strmatch(class,th14classnames,'exact');
  assert(length(classid)==1);

    [rec,prec,ap,bgcls_err,othercls_err,loc_err]=TH14eventdetpr(detevents,gtevents,class,threshold);
    pr_all(i,1).class=class;
    pr_all(i,1).classid=classid;
    pr_all(i,1).overlapthresh=threshold;
    pr_all(i,1).prec=prec;
    pr_all(i,1).rec=rec;
    pr_all(i,1).ap=ap;
    pr_all(i,1).bgcls_err=bgcls_err;
    pr_all(i,1).othercls_err=othercls_err;
    pr_all(i,1).loc_err=loc_err;
    ap_all(i,1)=ap;
    
    fprintf('AP:%1.3f at overlap %1.1f for %s\n',ap,threshold,class)
end

map=mean(ap_all,1);
ap_all=ap_all';
fprintf('\n\nMAP: %f \n\n',map);


function [rec,prec,ap,bgcls_err,othercls_err,loc_err]=TH14eventdetpr(detevents,gtevents,class,overlapthresh)

  
gtvideonames={gtevents.videoname};
detvideonames={detevents(:).videoname};
videonames=unique(cat(2,gtvideonames,detvideonames));

%tpconf=[];
%fpconf=[];
unsortConf=[];
unsortFlag=[];
npos=length(strmatch(class,{gtevents.class},'exact'));
assert(npos>0)

indgtclass=strmatch(class,{gtevents.class},'exact');
indambclass=strmatch('Ambiguous',{gtevents.class},'exact');
indothergtclass = setdiff([1:length(gtevents)], [indgtclass;indambclass]);
inddetclass=strmatch(class,{detevents.class},'exact');

if length(inddetclass)==0
    fprintf('Class %s no instance, skip\n',class);
    rec=0;
    prec=0;
    ap=0;
    return;
end

correctPortion=zeros(length(videonames),1);
groundNum=zeros(length(videonames),1);

for i=1:length(videonames)
  correctPortionName{i,1}=videonames{i};
  gt=gtevents(intersect(strmatch(videonames{i},gtvideonames,'exact'),indgtclass));
  othergt=gtevents(intersect(strmatch(videonames{i},gtvideonames,'exact'),indothergtclass));
  amb=gtevents(intersect(strmatch(videonames{i},gtvideonames,'exact'),indambclass)); 
  det=detevents(intersect(strmatch(videonames{i},detvideonames,'exact'),inddetclass));
  
  groundNum(i) = length(gt);
  
  if length(det)
  
    [vs,is]=sort(-[det(:).conf]);
    det=det(is);
    conf=[det(:).conf];
    indfree=ones(1,length(det));
    indamb=zeros(1,length(det));

    % interesct event detection intervals with GT
    if length(gt)
      ov=intervaloverlapvalseconds(cat(1,gt(:).timeinterval),cat(1,det(:).timeinterval));
      % add Loc_Err unmatched
      indfree(sum(ov,1) > 0) = 3;
      for k=1:size(ov,1)
        ind=find(indfree>0);
        [vm,im]=max(ov(k,ind));
        if vm>overlapthresh
            indfree(ind(im))=0;
        end
      end
    end
    
    % interesct event detection intervals with otherGT
    if length(othergt)
      otherov=intervaloverlapvalseconds(cat(1,othergt(:).timeinterval),cat(1,det(:).timeinterval));
      for k=1:size(otherov,1)
        ind=find(indfree==1);
        [vm,im]=max(otherov(k,ind));
        if vm>overlapthresh
            indfree(ind(im))=4;
        end
      end
    end
    
    % respect ambiguous events (overlapping detections will be removed from the FP list)
    if length(amb)
      ovamb=intervaloverlapvalseconds(cat(1,amb(:).timeinterval),cat(1,det(:).timeinterval));
      indamb=sum(ovamb,1);
    end
    
    idx1 = find(indfree==0); %matched
    idx2 = find(indfree==1 & indamb==0); %unmatched BGCls_ERR
    idx3 = find(indfree==3 & indamb==0); %unmatched Loc_ERR
    idx4 = find(indfree==4 & indamb==0); %unmatched OtherCls_ERR

    flag = [ones(size(idx1)) 2*ones(size(idx2)) 3*ones(size(idx3)) 4*ones(size(idx4))];
    [idxall, ttIdx] = sort([idx1 idx2 idx3 idx4]);
    flagall = flag(ttIdx);
    unsortConf = [unsortConf conf(idxall)];
    unsortFlag = [unsortFlag flagall];
    
    %tpconf=[tpconf conf(find(indfree==0))];
    %fpconf=[fpconf conf(find(indfree==1))];
    %fpconf=[fpconf conf(find(indfree==1 & indamb==0))];
    if length(gt)~=0
        correctPortion(i) = length(find(indfree==0))/length(gt);
    end
    
  end
end

%conf=[tpconf fpconf; 1*ones(size(tpconf)) 2*ones(size(fpconf))];
conf=[unsortConf; unsortFlag];

[vs,is]=sort(-conf(1,:));
tp=cumsum(conf(2,is)==1);
fp=cumsum(conf(2,is)==2 | conf(2,is)==3 | conf(2,is)==4);
tmp=conf(2,is)==1;
rec=tp/npos;
prec=tp./(fp+tp);
ap=prap(rec,prec,tmp,npos);
%Cls_Err and Loc_Err
fp_bgcls=cumsum(conf(2,is)==2);
fp_loc=cumsum(conf(2,is)==3);
fp_othercls=cumsum(conf(2,is)==4);

bgcls_err = fp_bgcls ./ (fp+tp) ;
loc_err = fp_loc ./ (fp+tp) ;
othercls_err = fp_othercls ./ (fp+tp) ;
%fprintf('correct: %1.3f,  cls_err: %1.3f, loc_err: %1.3f \n',correct,cls_err,loc_err);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ap=prap(rec,prec,tmp,npos)

ap=0;
for i=1:length(prec)
    if tmp(i)==1
        ap=ap+prec(i);
    end
end
ap=ap/npos;


