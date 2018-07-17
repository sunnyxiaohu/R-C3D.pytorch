function pick = nms_temporal(boxes, overlap)
% 
% top = nms(boxes, overlap)
% Non-maximum suppression. (FAST VERSION)
% Greedily select high-scoring detections and skip detections
% that are significantly covered by a previously selected
% detection.
%
% NOTE: This is adapted from Pedro Felzenszwalb's version (nms.m),
% but an inner loop has been eliminated to significantly speed it
% up in the case of a large number of boxes

% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm


if isempty(boxes)
  pick = [];
  return;
end

x1 = boxes(:,1);
x2 = boxes(:,2);
s = boxes(:,end);

union = x2 - x1;
[vals, I] = sort(s);

pick = s*0;
counter = 1;
while ~isempty(I)
  last = length(I);
  i = I(last);  
  pick(counter) = i;
  counter = counter + 1;
  
  xx1 = max(x1(i), x1(I(1:last-1)));
  xx2 = min(x2(i), x2(I(1:last-1)));
  
  w = max(0.0, xx2-xx1);
  
  inter = w;
  o = inter ./ (union(i) + union(I(1:last-1)) - inter);
  
  I = I(find(o<=overlap));
end

pick = pick(1:(counter-1));
