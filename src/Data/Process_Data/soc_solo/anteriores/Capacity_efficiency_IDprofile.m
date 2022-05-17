%% Reference model: ssc_lithium_cell_2RC
clear all; clc;

%% Load Real experimental test information
% load RealTestData;
load('Perfil_ID.mat');

% Calculate capacity at different temperatures for charge and discharge
% 25deg
Cbat_test_cha25 = abs(Perfil_ID(3).C_Ah(268408+2:287056,1));
Cbat_test_total_cha25 = max(Cbat_test_cha25)
Cbat_test_dch25 = abs(Perfil_ID(3).C_Ah(287057:336320,1));
Cbat_test_total_dch25 = sum(Cbat_test_dch25)/1000
eff_25 = (Cbat_test_total_dch25/Cbat_test_total_cha25)
Q25 = Cbat_test_total_dch25

% 45deg
Cbat_test_cha45 = abs(Perfil_ID(1).C_Ah(417833+2:435888,1));
Cbat_test_total_cha45 = max(Cbat_test_cha45)
Cbat_test_dch45 = abs(Perfil_ID(1).C_Ah(435889:485758,1));
Cbat_test_total_dch45 = sum(Cbat_test_dch45)/1000
eff_45 = (Cbat_test_total_dch45/Cbat_test_total_dch25)
Q45 = Q25*eff_45 
% eff_45 = (Cbat_test_total_dch45/Cbat_test_total_cha45) % eficiencia en el ciclo a 45deg
% eff_45 = (Cbat_test_total_dch45/Cbat_test_total_cha45)-eff_25*(Cbat_test_total_cha25/Cbat_test_total_cha45)

% 5deg
Cbat_test_cha5 = abs(Perfil_ID(5).C_Ah(268283+1:286217,1));
Cbat_test_total_cha5 = max(Cbat_test_cha5)
Cbat_test_dch5 = abs(Perfil_ID(5).C_Ah(286218:335619,1));
Cbat_test_total_dch5 = sum(Cbat_test_dch5)/1000
eff_5 = (Cbat_test_total_dch5/Cbat_test_total_dch25)
Q5 = Q25*eff_5 
% eff_5 = (Cbat_test_total_dch5/Cbat_test_total_cha5) % eficiencia en el ciclo a 5deg
% eff_5 = (Cbat_test_total_dch5/Cbat_test_total_cha5)-eff_25*(Cbat_test_total_cha25/Cbat_test_total_cha5)


