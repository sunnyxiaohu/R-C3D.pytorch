function target_intervals = transform_reg_target_inv(src_intervals, reg_label)
%input: bt_intervals, reg_label
%output: 
%t = (p_t x t_src) + t_src), l = l_src x exp(p_l)
%t* = (p_t* x t_src) + t_src), l* = l_src x exp(p_l*)
%where 

center_intervals_src = 0.5 * (src_intervals(:, 2) + src_intervals(:, 1));
len_intervals_src = src_intervals(:, 2) - src_intervals(:, 1);

t_target = reg_label(:, 1) .* center_intervals_src + center_intervals_src;
l_target =  len_intervals_src .* exp(reg_label(:,2));

target_intervals = [t_target - 0.5 * l_target, t_target + 0.5 * l_target];
end