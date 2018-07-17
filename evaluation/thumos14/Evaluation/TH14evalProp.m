function [rec_all,prec_all]=TH14evalProp(detfilename,gtpath,subset,threshold)
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
    gtevents(gteventscount).class='Agnostic';%class;
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
  %ind=find(clsid(i)==th14classids);
  ind=find(clsid(i)==100);
  if length(ind)
    detevents(i).videoname=videonames{i};
    detevents(i).timeinterval=[t1(i) t2(i)];
    detevents(i).class= 'Agnostic';%th14classnames{ind};
    detevents(i).conf=conf(i);
  else
    fprintf('WARNING: Reported class ID %d is not among THUMOS14 detection classes.\n')
  end
end

% Visualize 
%
show_data = false;
ap_all=[];
clear pr_all

class='Agnostic'; %th14classnames{i};
classid= 100;%strmatch(class,th14classnames,'exact');
assert(length(classid)==1);

[rec_all,prec_all,ap]=TH14eventdetpr(detevents,gtevents,class,threshold,show_data);


fprintf('Recall:%1.3f at overlap %1.2f for %s\n',rec_all(end),threshold,class);

fprintf('Precision:%1.3f at overlap %1.2f for %s\n',prec_all(end),threshold,class);



function [rec,prec,ap]=TH14eventdetpr(detevents,gtevents,class,overlapthresh,show_data)

  
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
  amb=gtevents(intersect(strmatch(videonames{i},gtvideonames,'exact'),indambclass)); 
  det=detevents(intersect(strmatch(videonames{i},detvideonames,'exact'),inddetclass));

  if show_data
     figure(1);clf;
     title(videonames{i})
     % plot all gt using red color
     for m=1:length(gt)     
         rectangle('Position',[gt(m).timeinterval(1),0.2*m,gt(m).timeinterval(2)-gt(m).timeinterval(1),0.2],...
             'FaceColor', 'r'); %[x,y,w,h]
     end
     % plot all det using green color
     for j=1:length(det)     
         rectangle('Position',[det(j).timeinterval(1),0.2*(m+j),det(j).timeinterval(2)-det(j).timeinterval(1),0.2],...
             'FaceColor', 'g'); %[x,y,w,h] 
     end
     % plot all dmb using yellow clor
     for k=1:length(amb)     
         rectangle('Position',[amb(j).timeinterval(1),0.2*(m+j+k),amb(k).timeinterval(2)-amb(k).timeinterval(1),0.2],...
             'FaceColor', 'y'); %[x,y,w,h] 
     end
     
  end
  
  
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
      for k=1:size(ov,1)
        ind=find(indfree);
        [vm,im]=max(ov(k,ind));
        if vm>overlapthresh
            indfree(ind(im))=0;
        end
      end
    end
    
    % respect ambiguous events (overlapping detections will be removed from the FP list)
    if length(amb)
      ovamb=intervaloverlapvalseconds(cat(1,amb(:).timeinterval),cat(1,det(:).timeinterval));
      indamb=sum(ovamb,1);
    end
    
    idx1 = find(indfree==0);
    idx2 = find(indfree==1 & indamb==0);
    flag = [ones(size(idx1)) 2*ones(size(idx2))];
    [idxall, ttIdx] = sort([idx1 idx2]);
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
fp=cumsum(conf(2,is)==2);
tmp=conf(2,is)==1;
rec=tp/npos;
prec=tp./(fp+tp);
ap=prap(rec,prec,tmp,npos);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ap=prap(rec,prec,tmp,npos)

ap=0;
for i=1:length(prec)
    if tmp(i)==1
        ap=ap+prec(i);
    end
end
ap=ap/npos;


