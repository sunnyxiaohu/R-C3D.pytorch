function video_proposals_dst = temporal_act_grouping(video_proposals_src, gama_list)
%input: 
%video_proposals_src : nx3 ,[start_t end_t confidence]
%video_proposals_dst : nx3, [start_t end_t confience]

    % plot all det using green color    
%     clf reset;
%     figure(1);
%     for j=1:length(video_proposals_src)     
%         rectangle('Position',[video_proposals_src(j,1),0.4*(j),video_proposals_src(j,2)-video_proposals_src(j,1),0.4],...
%          'FaceColor', 'g'); %[x,y,w,h] 
%         text(video_proposals_src(j,2),0.4*(j+1), num2str(video_proposals_src(j,3)));
%     end
    
    % generate 1D score curve
    time_seg = [video_proposals_src(:,1) ; video_proposals_src(:,2)];
    [~, order] = sort(time_seg);
    time_seg = time_seg(order);
    seg_key = zeros(length(time_seg)+1,1); seg_value = zeros(length(time_seg)+1,1);
    seg_key(1) = time_seg(1); seg_key(end) = time_seg(end);
    for t=1:length(time_seg)-1
        seg_key(t+1) =  0.5 * (time_seg(t) + time_seg(t+1));
        ind = find(video_proposals_src(:,1)<=time_seg(t) & video_proposals_src(:,2)>=time_seg(t+1));
        if ~isempty(ind)
            seg_value(t+1) = mean(video_proposals_src(ind,3));
        end
    end
%     figure(2);
%     plot(seg_key, seg_value);
    
    % grouping proposals by gama_list
    start_flag = true;
    start_t = []; end_t = []; conf_t = [];
    for gama=gama_list
        for t=1:length(seg_key)
            if seg_value(t)>=gama && start_flag==true
                start_t = [start_t ; seg_key(t)];
                start_flag = false;
            end
            if seg_value(t)<gama && start_flag == false
                end_t = [end_t ; seg_key(t)];
                conf_t = [conf_t; gama];
                start_flag = true;
            end
        end
    end
    video_proposals_dst = [start_t, end_t, conf_t];
%     hold on;
%     for i=1:size(video_proposals_dst,1)
%         %start time
%         plot([video_proposals_dst(i,1) video_proposals_dst(i,1)], [0 video_proposals_dst(i,3)], '--r');
%         %end time
%         plot([video_proposals_dst(i,2) video_proposals_dst(i,2)], [0 video_proposals_dst(i,3)], '--r');
%     end
%     hold off;
    
end