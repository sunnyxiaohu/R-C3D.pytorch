function ov=intervaloverlapvalseconds(i1,i2,normtype,gt,det)
 %

 
if nargin<3 normtype=0; end
  
ov=zeros(size(i1,1),size(i2,1));
for i=1:size(i1,1) 
  for j=1:size(i2,1) 
    ov(i,j)=intervalsingleoverlapvalseconds(i1(i,:),i2(j,:),normtype);
    if nargin==5 
        ov(i,j)=ov(i,j)*strcmp(gt(i).class,det(j).class);
    end
  end
end

function ov=intervalsingleoverlapvalseconds(i1,i2,normtype)

 
  
i1=[min(i1) max(i1)];
i2=[min(i2) max(i2)];

ov=0;
if normtype<0 ua=1;
elseif normtype==1
  ua=(i1(2)-i1(1));
elseif normtype==2
  ua=(i2(2)-i2(1));
else
  bu=[min(i1(1),i2(1)) ; max(i1(2),i2(2))];
  ua=(bu(2)-bu(1));
end

bi=[max(i1(1),i2(1)) ; min(i1(2),i2(2))];
iw=bi(2)-bi(1);
if iw>0
  if normtype<0 % no normalization!
    ov=iw;
  else
    ov=iw/ua;
  end
end


%i1=i1(:)';
%i2=i2(:)';

%ov=0;
%[vs,is]=sort([i1(1:2) i2(1:2)]);
%ind=[1 1 2 2];
%inds=ind(is);
%if inds(1)~=inds(2) 
%  ov=vs(3)-vs(2); 
%end
