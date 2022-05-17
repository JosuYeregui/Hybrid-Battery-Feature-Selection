%% Reference model: ssc_lithium_cell_2RC
clear all; clc;

%% Load Real experimental test information
load('Perfil_ID.mat'); % Dynamic profiles
load('SOC_OCV.mat'); % OCV files
Rconv = 6.15; Rcond = 0; Cth = 25.95; % Thermal parameters

% Seleccionar el test que se necesita
% i = 1 --> 5 deg
% i = 2 --> 25 deg
% i = 3 --> 45 deg
i = 1;

if i == 1
%% Perfil_ID: 5deg
% Calculo de capacidad y eficiencia a 25 deg (temperatura de referencia)
Time_test = Perfil_ID(3).t_h(287057:336320,1)-min(Perfil_ID(3).t_h(287057:336320,1));
Ibat_test = Perfil_ID(3).I_A(287057:336320,1);
Integracion_Ibat = abs(sum(Ibat_test.*[0;diff(Time_test)])/3600);
Q25 = Integracion_Ibat; % Discharge capacity at 25deg
clear Ibat_test Integracion_Ibat Time_test

% 5 deg
Time_test = Perfil_ID(5).t_h(286218:335619,1)-min(Perfil_ID(5).t_h(286218:335619,1));
Vbat_test = Perfil_ID(5).V_V(286218:335619,1);
Ibat_test = Perfil_ID(5).I_A(286218:335619,1);
Tbat_test = Perfil_ID(5).T_deg(286218:335619,1)+273.15;

% Calculo de capacidad y eficiencia a 5 deg 
Integracion_Ibat = abs(sum(Ibat_test.*[0;diff(Time_test)])/3600);
Q5 = Integracion_Ibat; % Discharge capacity at 25deg
eff_5 = (Q25/Q5);

% % Condiciones iniciales a 5deg
T_init = 273.15 + 5 + 1; % K % Offset aplicado a 25deg
Qnom = Q25/eff_5;
Q0 = Q25/eff_5;
t_sim_end = Time_test(end,1);

% Inicialización parámetros OCV 5 deg
SOC_full = flip(SOC_OCVreal(:,1)*100);
SOC_LUT = linspace(SOC_full(1), SOC_full(end),10);
SOC_LUT(1,1) = 5; % El primer punto de SOC no está bien definido por el experimental
Em_full = flip(SOC_OCVreal(:,2));
Em_LUT = interp1(SOC_full,Em_full,SOC_LUT);

% Inicialización de parámetros R-Cs (ajustados a 25deg)
C1_LUT = [235.17 1246 1252.7 1261.4 1256.6 1258.7 1259.1 1259.8 1262.9 1235.3];
C2_LUT = [90.935 3348.7 3590.5 3607.1 3645.6 3544.4 3532 3604.4 3591 4517.6];
Em_LUT = [3.1658 3.4768 3.5975 3.7303 3.7706 3.7882 3.8381 3.9462 4.0582 4.1668];
R0_LUT = [0.0030829 0.072628 0.059416 0.043535 0.050516 0.04424 0.044157 0.042746 0.040859 0.035273];
R1_LUT = [0.0015586 0.0052784 0.0076522 0.0073713 0.0040773 0.007966 0.0042052 0.0083473 0.0082109 0.0015675];
R2_LUT = [0.010561 0.0057407 0.0077463 0.0090309 0.015808 0.0092643 0.0066197 0.030052 0.010592 0.0063853];

% Guardar valores de los parámetros en un lugar común para todas las Temperaturas
Em_LUT_T(1,:) = Em_LUT;
R0_LUT_T(1,:) = R0_LUT;
C1_LUT_T(1,:) = C1_LUT;
C2_LUT_T(1,:) = C2_LUT;
R1_LUT_T(1,:) = R1_LUT;
R2_LUT_T(1,:) = R2_LUT;

elseif i==2
%% Perfil_ID: 25 deg
% Datos necesarios para comparar la simulación con el experimental 
Time_test = Perfil_ID(3).t_h(287057:336320,1)-min(Perfil_ID(3).t_h(287057:336320,1));
Vbat_test = Perfil_ID(3).V_V(287057:336320,1);
Ibat_test = Perfil_ID(3).I_A(287057:336320,1);
Tbat_test = Perfil_ID(3).T_deg(287057:336320,1)+273.15;

% Calculo de capacidad y eficiencia a 25deg
Integracion_Ibat = abs(sum(Ibat_test.*[0;diff(Time_test)])/3600);
Q25 = Integracion_Ibat; % Discharge capacity at 25deg

% Condiciones iniciales a 25deg
T_init = 273.15 + 25 +0.2; % K % Offset aplicado a 25deg
Qnom = Q25;
Q0 = Q25;
t_sim_end = Time_test(end,1);

% Inicialización parámetros OCV 25 deg
SOC_full = flip(SOC_OCVreal(:,1)*100);
SOC_LUT = linspace(SOC_full(1), SOC_full(end),10);
SOC_LUT(1,1) = 5; % El primer punto de SOC no está bien definido por el experimental
Em_full = flip(SOC_OCVreal(:,3));
Em_LUT = interp1(SOC_full,Em_full,SOC_LUT);

% Inicialización de parámetros R-Cs (ajustados a 25 deg, sin multiobj de T)
C1_LUT = [235.17 1246 1252.7 1261.4 1256.6 1258.7 1259.1 1259.8 1262.9 1235.3];
C2_LUT = [90.935 3348.7 3590.5 3607.1 3645.6 3544.4 3532 3604.4 3591 4517.6];
Em_LUT = [3.1658 3.4768 3.5975 3.7303 3.7706 3.7882 3.8381 3.9462 4.0582 4.1668];
R0_LUT = [0.0030829 0.072628 0.059416 0.043535 0.050516 0.04424 0.044157 0.042746 0.040859 0.035273];
R1_LUT = [0.0015586 0.0052784 0.0076522 0.0073713 0.0040773 0.007966 0.0042052 0.0083473 0.0082109 0.0015675];
R2_LUT = [0.010561 0.0057407 0.0077463 0.0090309 0.015808 0.0092643 0.0066197 0.030052 0.010592 0.0063853];
% Rconv = 9.9174

% Guardar valores de los parámetros en un lugar común para todas las Temperaturas
Em_LUT_T(2,:) = Em_LUT;
R0_LUT_T(2,:) = R0_LUT;
C1_LUT_T(2,:) = C1_LUT;
C2_LUT_T(2,:) = C2_LUT;
R1_LUT_T(2,:) = R1_LUT;
R2_LUT_T(2,:) = R2_LUT;

else
%%  Perfil_ID: 45deg
% Calculo de capacidad y eficiencia a 25 deg (temperatura de referencia)
Time_test = Perfil_ID(3).t_h(287057:336320,1)-min(Perfil_ID(3).t_h(287057:336320,1));
Ibat_test = Perfil_ID(3).I_A(287057:336320,1);
Integracion_Ibat = abs(sum(Ibat_test.*[0;diff(Time_test)])/3600);
Q25 = Integracion_Ibat; % Discharge capacity at 25deg
clear Ibat_test Integracion_Ibat Time_test

% 45 deg
Time_test = Perfil_ID(1).t_h(435889:485758-3,1)-min(Perfil_ID(1).t_h(435889:485758-3,1));
Vbat_test = Perfil_ID(1).V_V(435889:485758-3,1);
Ibat_test = Perfil_ID(1).I_A(435889:485758-3,1);
Tbat_test = Perfil_ID(1).T_deg(435889:485758-3,1)+273.15;

% Calculo de capacidad y eficiencia a 45 deg 
Integracion_Ibat = abs(sum(Ibat_test.*[0;diff(Time_test)])/3600);
Q45 = Integracion_Ibat; % Discharge capacity at 25deg
eff_45 = (Q25/Q45);

% % Condiciones iniciales a 45deg
T_init = 273.15 + 45 -1; % K % Offset aplicado a 25deg
Qnom = Q25/eff_45;
Q0 = Q25/eff_45;
t_sim_end = Time_test(end,1);

% Inicialización parámetros OCV 45 deg
SOC_full = flip(SOC_OCVreal(:,1)*100);
SOC_LUT = linspace(SOC_full(1), SOC_full(end),10);
SOC_LUT(1,1) = 5; % El primer punto de SOC no está bien definido por el experimental
Em_full = flip(SOC_OCVreal(:,4));
Em_LUT = interp1(SOC_full,Em_full,SOC_LUT);

% Inicialización de parámetros R-Cs (ajustados a 25deg)
C1_LUT = [235.17 1246 1252.7 1261.4 1256.6 1258.7 1259.1 1259.8 1262.9 1235.3];
C2_LUT = [90.935 3348.7 3590.5 3607.1 3645.6 3544.4 3532 3604.4 3591 4517.6];
Em_LUT = [3.1658 3.4768 3.5975 3.7303 3.7706 3.7882 3.8381 3.9462 4.0582 4.1668];
R0_LUT = [0.0030829 0.072628 0.059416 0.043535 0.050516 0.04424 0.044157 0.042746 0.040859 0.035273];
R1_LUT = [0.0015586 0.0052784 0.0076522 0.0073713 0.0040773 0.007966 0.0042052 0.0083473 0.0082109 0.0015675];
R2_LUT = [0.010561 0.0057407 0.0077463 0.0090309 0.015808 0.0092643 0.0066197 0.030052 0.010592 0.0063853];

% Guardar valores de los parámetros en un lugar común para todas las Temperaturas
Em_LUT_T(3,:) = Em_LUT;
R0_LUT_T(3,:) = R0_LUT;
C1_LUT_T(3,:) = C1_LUT;
C2_LUT_T(3,:) = C2_LUT;
R1_LUT_T(3,:) = R1_LUT;
R2_LUT_T(3,:) = R2_LUT;
end


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