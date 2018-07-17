% variable: ov
clear all; clf;
sigma = 0.5;
ov = 0:0.1:1;

figure(1);
hold on;
for score=0.3:0.3:1
    f = score .* exp(ov.^2 ./sigma);
    plot(ov, f);
end
hold off;
title('Adjust score along overlap, with score fixed');
