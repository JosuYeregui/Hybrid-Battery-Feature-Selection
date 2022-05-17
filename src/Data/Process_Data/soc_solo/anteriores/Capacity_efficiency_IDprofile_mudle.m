%% Reference model: ssc_lithium_cell_2RC
clear all; clc;

%% Load Real experimental test information
% load RealTestData;
load('Perfil_ID.mat');

% Calculate capacity at different temperatures for charge and discharge
% 25deg
% Cbat_test_cha25 = abs(Perfil_ID(3).C_Ah(268408+2:287056,1));
% Cbat_test_total_cha25 = max(Cbat_test_cha25)
% Cbat_test_dch25 = abs(Perfil_ID(3).C_Ah(287057:336320,1));
% Cbat_test_total_dch25 = sum(Cbat_test_dch25)/1000
% eff_25 = (Cbat_test_total_dch25/Cbat_test_total_cha25)


% [Time_test,Vbat_test]
% [Time_test,Tbat_test]
% [Time_test,Ibat_test]

Time_test = Perfil_ID(3).t_h(287057:336320,1)-min(Perfil_ID(3).t_h(287057:336320,1));
Vbat_test = Perfil_ID(3).V_V(287057:336320,1);
Ibat_test = Perfil_ID(3).I_A(287057:336320,1);
Integracion_Ibat = abs(sum(Ibat_test.*[0;diff(Time_test)])/3600)
Q25 = Integracion_Ibat
Tbat_test = Perfil_ID(3).T_deg(287057:336320,1)+273.15;
figure()
plot(Time_test,Vbat_test)
% figure()
% plot(Time_test,Cbat_test_dch25)

% % 45deg
% Cbat_test_cha45 = abs(Perfil_ID(1).C_Ah(417833+2:435888,1));
% Cbat_test_total_cha45 = max(Cbat_test_cha45)
% Cbat_test_dch45 = abs(Perfil_ID(1).C_Ah(435889:485758,1));
% Cbat_test_total_dch45 = sum(Cbat_test_dch45)/1000
% eff_45 = (Cbat_test_total_dch45/Cbat_test_total_dch25)
% Q45 = Q25*eff_45 
% 
% Time_test = Perfil_ID(1).t_h(435889:485758,1)-min(Perfil_ID(3).t_h(435889:485758,1));
% Vbat_test = Perfil_ID(1).V_V(435889:485758,1);
% Ibat_test = Perfil_ID(1).I_A(435889:485758,1);
% Tbat_test = Perfil_ID(1).T_deg(435889:485758,1)+273.15;
% figure()
% plot(Time_test,Vbat_test)
% figure()
% plot(Time_test,Cbat_test_dch45)
% 
% % 5deg
% Cbat_test_cha5 = abs(Perfil_ID(5).C_Ah(268283+1:286217,1));
% Cbat_test_total_cha5 = max(Cbat_test_cha5)
% Cbat_test_dch5 = abs(Perfil_ID(5).C_Ah(286218:335619,1));
% Cbat_test_total_dch5 = sum(Cbat_test_dch5)/1000
% eff_5 = (Cbat_test_total_dch5/Cbat_test_total_dch25)
% Q5 = Q25*eff_5 
% 
% Time_test = Perfil_ID(5).t_h(286218:335619,1)-min(Perfil_ID(3).t_h(286218:335619,1));
% Vbat_test = Perfil_ID(5).V_V(286218:335619,1);
% Ibat_test = Perfil_ID(5).I_A(286218:335619,1);
% Tbat_test = Perfil_ID(5).T_deg(286218:335619,1)+273.15;
% figure()
% plot(Time_test,Vbat_test)
% figure()
% plot(Time_test,Cbat_test_dch5)

% Load OCV files
load('SOC_OCV.mat');

%% Initial and nominal capacities
T_init = 273.15 + 25 +0.2; % K % EJEMPLO
Rconv = 6.15;
Rcond = 0;
Cth = 25.95;
Qnom = Q25;
Q0 = Q25;

%% Inicialización del modelo
SIM_Ibat = Ibat_test;
SIM_Tbat = Tbat_test;
SIM_time = Time_test;
SIM_Vbat = Vbat_test;

% SOC_full = flip(SOC_OCVreal(:,1)*100);
% SOC_LUT = linspace(SOC_full(1), SOC_full(end),10);
% Em_full = flip(SOC_OCVreal(:,4));
% Em_LUT = interp1(SOC_full,Em_full,SOC_LUT);
% figure()
% plot(SOC_full,Em_full)
% hold on
% plot(SOC_LUT, Em_LUT)

% R0_LUT = [0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024];
% R1_LUT = [0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012];
% R2_LUT = [0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073];
% C1_LUT = [1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3];
% C2_LUT = [3.55e3 3.55e3 3.55e3 3.55e3 3.55e3 3.55e3 3.55e3 3.55e3 3.55e3 3.55e3];

% Iteración 1 se ha parado
% C1_LUT = [1272.3 1272.3 1272.3 1272.3 1272.3 1272.3 1272.3 1272.3 1272.3 1272.3]
% C2_LUT = [3550 3550 3550 3550 3550 3550 3550 3550 3550 3550]
% R0_LUT = [0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024]
% R1_LUT = [0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012]
% R2_LUT = [0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073]

% C1_LUT = [1272.3 1272.3 1272.3 1272.3 1272.3 1272.3 1272.3 1272.3 1272.3 1272.3]
% C2_LUT = [3550 3550 3550 3550 3550 3550 3550 3550 3550 3550]
% R0_LUT = [0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024]
% R1_LUT = [0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012]
% R2_LUT = [0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073]

% SOC_LUT = [5 11.1111111111111	22.2222222222222	33.3333333333333	44.4444444444444	55.5555555555556	66.6666666666667	77.7777777777778	88.8888888888889	100]
% C1_LUT = [1268.9 1268.9 1267.4 1268.8 1267.7 1267.7 1267.4 1267.9 1267.9 1267.5]
% C2_LUT = [3537.9 3537.9 3550 3551.1 3553.1 3548.5 3548.2 3548.9 3548.6 3551.6]
% Em_LUT = [3 3.4611 3.5889 3.7163 3.7658 3.7844 3.8379 3.9354 4.0516 4.1757]
% R0_LUT = [0.0024017 0.0048149 0.0042337 0.0038845 0.0040293 0.0038935 0.0037758 0.003829 0.0037295 0.0036823]
% R1_LUT = [0.0012009 0.0017978 0.0019296 0.0016301 0.0015758 0.0016153 0.0016457 0.0016761 0.0016765 0.0014734]
% R2_LUT = [0.0073061 0.00729 0.0073064 0.0073106 0.0075407 0.0077706 0.0068437 0.0086929 0.0079428 0.0078584]
% Rconv = 9.9174

SOC_LUT = [5 11.1111111111111	22.2222222222222	33.3333333333333	44.4444444444444	55.5555555555556	66.6666666666667	77.7777777777778	88.8888888888889	100]
C1_LUT = [1268.9 1268.9 1267.4 1268.8 1267.7 1267.7 1267.4 1267.9 1267.9 1267.5]
C2_LUT = [3537.9 3537.9 3550 3551.1 3553.1 3548.5 3548.2 3548.9 3548.6 3551.6]
Em_LUT = [3 3.4611 3.5889 3.7163 3.7658 3.7844 3.8379 3.9354 4.0516 4.1757]
R0_LUT = [0.0024017 0.0048149 0.0042337 0.0038845 0.0040293 0.0038935 0.0037758 0.003829 0.0037295 0.0036823]
R1_LUT = [0.0012009 0.0017978 0.0019296 0.0016301 0.0015758 0.0016153 0.0016457 0.0016761 0.0016765 0.0014734]
R2_LUT = [0.0073061 0.00729 0.0073064 0.0073106 0.0075407 0.0077706 0.0068437 0.0086929 0.0079428 0.0078584]
Rconv = 9.9174

%% Simulación 25ºC ronda 1 (sin optimizar OCV)
%    C1_LUT = [1419.6 1255.9 1293.6 1290.2 1279.1 1269.4 1273.9 1277.7 1273.7 1270.4];
%    C2_LUT = [4228.4 3507.1 3646.1 3592.3 3563.1 3561.1 3568.1 3571.6 3556.6 3542.7];
%    R0_LUT = [0.0023862 0.0024418 0.0024462 0.0024577 0.0024645 0.0024545 0.0024547 0.0024507 0.002453 0.002455];
%    R1_LUT = [0.0014414 0.0011817 0.0012402 0.0012183 0.001211 0.0012118 0.0012144 0.001216 0.0012098 0.0012051];
%    R2_LUT = [0.0038121 0.00619 0.0066879 0.0070016 0.0072586 0.0071822 0.0071793 0.0071287 0.0072553 0.0073786];
%     Rconv = 6.7335
%     T_init = 298.48
%     Em_LUT = [2.7 3.4006 3.5693 3.7031 3.7668 3.7907 3.8451 3.9365 4.0609 4.1992]

% Optimizando OCV
% C1_LUT = [1514.2 1600.6 1164.6 1268.7 1240 1153 1214 1247.7 1183.8 1230.6]
% C2_LUT = [65269 2430.7 3520.2 3758.4 3516.6 3498.6 3563 3547.5 3448.4 3435.8]
% R0_LUT = [0.0054848 0.0043156 0.004004 0.0040177 0.0040842 0.0039195 0.0039268 0.0038573 0.0037982 0.0037903]
% R1_LUT = [0.0049815 0.00067725 0.0014304 0.0014336 0.0013513 0.0013636 0.0013953 0.0014284 0.0013613 0.0013323]
% R2_LUT = [0.0030768 0.0050176 0.0087063 0.0064877 0.0085634 0.007739 0.0081102 0.0082886 0.0083975 0.0087802]
%     Rconv = 8.9865
%     T_init = 298.14
%     Em_LUT = [3.3663 3.4509 3.6607 3.7393 3.7713 3.8021 3.8587 3.9552 4.0665 4.1681]

%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% % Resultados Marlon 
% % Valores ya optimizados 5deg
% Em_LUT = [3.3522 3.5128 3.6335 3.7354 3.7767 3.7861 3.8208 3.8767 3.9742 4.0681 4.1753];
% R0_LUT = [0.24479 0.20681 0.20011 0.17832 0.15846 0.15285 0.14252 0.13364 0.13031 0.12465 0.13];
% C1_LUT = [5255.5 164.76 617.04 184.36 929.05 969.67 1058.7 1313.6 867.05 592.25 1659.7];
% C2_LUT = [565.34 13153 46819 3699.2 5517.6 6496.9 5519 1.0447e+05 487.15 41264 27200];
% R1_LUT = [0.099177 0.11193 0.16118 0.031126 0.047102 0.035774 0.030818 0.066266 0.11205 0.060159 0.11638];
% R2_LUT = [0.30343 0.035323 0.074524 0.18435 0.12049 0.052321 0.064893 0.015627 0.0044847 0.01576 0.077592];
% %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% Em_LUT_T(1,:) = Em_LUT;
% R0_LUT_T(1,:) = R0_LUT;
% C1_LUT_T(1,:) = C1_LUT;
% C2_LUT_T(1,:) = C2_LUT;
% R1_LUT_T(1,:) = R1_LUT;
% R2_LUT_T(1,:) = R2_LUT;

% % Para 25º
%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% Valores ya optimizados
% Em_LUT = [3.4903 3.4903 3.6092 3.7189 3.7709 3.783 3.8216 3.8791 3.981 4.0801 4.1866];
% R0_LUT = [0.22762 0.16436 0.15389 0.14817 0.13523 0.12485 0.11501 0.10524 0.1203 0.11204 0.10405];
% C1_LUT = [64063 6.3166 2348.3 15576 2175.1 14885 2344.9 2549.7 5829.6 1974.2 3285.8];
% C2_LUT = [346.21 6031.6 5.4519e+05 3250.1 5924.5 454.29 9503.2 13704 17047 2.2501e+05 21653];
% R1_LUT = [1.6252 0.072549 0.054579 0.0079848 0.0090198 0.016061 0.016075 0.035818 0.030791 0.038739 0.043502];
% R2_LUT = [0.03976 0.11537 0.0047511 0.057196 0.042698 0.020021 0.033509 0.0061142 0.0012266 0.0525 0.39845];
%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% Em_LUT_T(2,:) = Em_LUT;
% R0_LUT_T(2,:) = R0_LUT;
% C1_LUT_T(2,:) = C1_LUT;
% C2_LUT_T(2,:) = C2_LUT;
% R1_LUT_T(2,:) = R1_LUT;
% R2_LUT_T(2,:) = R2_LUT;

% % Para 45º
%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% Valores ya optimizados
% Em_LUT = [3.4000 3.5287 3.6534 3.7318 3.7704 3.7893 3.8324 3.8916 3.9908 4.0844 4.1905];
% R0_LUT = [0.0540 0.0480 0.0438 0.0317 0.0325 0.0329 0.0342 0.0254 0.0375 0.0361 0.0410];
% C1_LUT = [6.20318919889900,7.09589219557527,4059.58358922060,76513.0881098971,8896.46953823830,43525.5179843206,5374.83055730033,226.575297831964,9912.92924585645,3436.35169010278,5089.82031551092];
% C2_LUT = [1556.77780906507,27931.8082304412,650033.918747548,3986.73953051236,13609.6994460913,1273.52154276700,17551.3278342163,127591.333368148,93472.9729969615,501940.987719753,10622.1751601134];
% R1_LUT = [0.0684 0.0660 0.0520 0.0075 0.0085 0.0146 0.0155 0.0346 0.0284 0.0363 0.0424];
% R2_LUT = [0.1886 0.1134 0.0046 0.0524 0.0382 0.0185 0.0318 0.0059 0.0012 0.0495 2.1401];
% %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% Em_LUT_T(3,:) = Em_LUT;
% R0_LUT_T(3,:) = R0_LUT;
% C1_LUT_T(3,:) = C1_LUT;
% C2_LUT_T(3,:) = C2_LUT;
% R1_LUT_T(3,:) = R1_LUT;
% R2_LUT_T(3,:) = R2_LUT;