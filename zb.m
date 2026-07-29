clear all,clc,close all;
load('./results/f_res_spark_ablation_w2_one.mat')
load('./results/f_res_spark_ablation_wlp_one.mat')
load('./grappa_recons/filledMask.mat')

% filledMask = rot90(rot90(logical(label_mask)));
% k空间信息(多通道)
kspace = permute(kspace, [2 3 1]);
res_spark_wdp_kspace = permute(res_spark_wlp_kspace, [2 3 1]);
res_spark_w2_kspace = permute(res_spark_w2_kspace, [2 3 1]);
[M,N,C] = size(kspace);

%% 合成k空间
mask = 150;
MCR_SPARK2_kspace = res_spark_w2_kspace;
MCR_SPARK2_kspace(M/2-mask/2:M/2+mask/2-1,N/2-mask/2:N/2+mask/2-1,:) = res_spark_wdp_kspace(M/2-mask/2:M/2+mask/2-1,N/2-mask/2:N/2+mask/2-1,:);

%% 替换ACS区域
mask = 30;
acsregionY_start = N /2-mask/2;% 计算acsregionY，这里直接使用MATLAB的索引方式
acsregionY_end = N/2+mask/2-1; % 减1是因为MATLAB的索引是从1开始
acsregionY = acsregionY_start:acsregionY_end;
res_spark_wdp_kspace(:, acsregionY, :) = kspace(:, acsregionY, :);
res_spark_w2_kspace(:, acsregionY, :) = kspace(:, acsregionY, :);
MCR_SPARK2_kspace(:, acsregionY, :) = kspace(:, acsregionY, :);

%% 图像域信息
truth = mifft2(kspace); %%多通道图像
res_spark_wdp = mifft2(res_spark_wdp_kspace);
res_spark_w2 = mifft2(res_spark_w2_kspace);
MCR_SPARK2 = mifft2(MCR_SPARK2_kspace);

truth_sos = sos(truth); %%单通道
res_spark_wdp_sos = sos(res_spark_wdp);
res_spark_w2_sos = sos(res_spark_w2);
MCR_SPARK2_sos = sos(MCR_SPARK2);

truth_brain = truth_sos.*filledMask; %%抠出脑袋区域
grappa_brain = grappa_sos.*filledMask;
spark_brain = spark_sos.*filledMask;
MCR_SPARK2_brain = MCR_SPARK2_sos.*filledMask;

figure();
subplot(1,2,1),imshow(truth_sos);title('truth');
subplot(1,2,2),imshow(MCR_SPARK2_sos);title('MCR-SPARK2');colormap('gray');