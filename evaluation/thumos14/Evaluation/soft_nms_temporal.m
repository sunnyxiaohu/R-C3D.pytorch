function pick = soft_nms_temporal(boxes, varargin)
% Apply soft nms for the temporal boxes.
    ip = inputParser;
    % The required input boxes with a 2-dim array, 
    % where boxes(i, :) = [x1, x2, s]
    ip.addRequired('boxes', @ismatrix);
    % The parameter of gaussian method.
    ip.addParamValue('sigma', 0.5, @isscalar);
    % The parameter of supression method.
    ip.addParamValue('Nt', 0.4, @isscalar);
    % It will remove the boxes, if its socre < threshold.
    ip.addParamValue('threshold', -Inf, @isscalar);
    % Specify the supression methods.
    ip.addParamValue('method', 0, @isscalar);
    ip.parse(boxes, varargin{:});
    result = ip.Results;
    
    if isempty(result.boxes)
        pick = [];
        return;
    end
    boxes = result.boxes;
    N = size(boxes, 1);
    pos = 0; maxpos = 0; maxscore = 0.0; 
    for i = 1:N
        maxscore = boxes(i,3);
        maxpos = i;
        tx1 = boxes(i,1);
        tx2 = boxes(i,2);
        ts = boxes(i,3);
        
        pos = i+1;
        % Get max box.
        while pos <= N
            if maxscore < boxes(pos,3);
                maxscore = boxes(pos,3);
                maxpos = pos;
            end
            pos = pos + 1;
        end
        
        % Add max box as a detection
        boxes(i,1) = boxes(maxpos, 1);
        boxes(i,2) = boxes(maxpos, 2);
        boxes(i,3) = boxes(maxpos, 3);
        % Swap ith box with position of max box
        boxes(maxpos,1) = tx1;
        boxes(maxpos,2) = tx2;
        boxes(maxpos,3) = ts;
        
        tx1 = boxes(i,1);
        tx2 = boxes(i,2);
        ts = boxes(i,3);
        
        pos = i + 1;
        
        % NMS iterations, note that N changes if dets fall below threshold.
        while pos <= N
            x1 = boxes(pos, 1);
            x2 = boxes(pos, 2);
            s = boxes(pos, 3);

            xx1 = max(tx1, x1);
            xx2 = min(tx2, x2);
            inter = max(0.0, xx2-xx1+1);
            ov = inter / (x2-x1+1 + tx2-tx1+1 -inter);
            % linear
            if result.method == 1
                if ov > result.Nt 
                    weight = 1 - ov;
                else
                    weight = 1;
                end
            % gaussian
            elseif result.method == 2
                weight = exp(-(ov * ov)/result.sigma);
            % original NMS
            else
                if ov > result.Nt
                    weight = 0;
                else
                    weight = 1;
                end
            end
            boxes(pos, 3) = weight*boxes(pos, 3);
		    
	        % if box score falls below threshold, discard the box 
            % by swapping with last box update N
            if boxes(pos, 3) < result.threshold
                boxes(pos,1) = boxes(N-1, 1);
                boxes(pos,2) = boxes(N-1, 2);
                boxes(pos,3) = boxes(N-1, 3);
                N = N - 1;
                pos = pos - 1;
            end
            pos = pos + 1;
        end
    end
    
    pick = 1:N;
end