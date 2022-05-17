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
i = 3;

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
C1_LUT = [3.0503 1034.8 1194.5 1213.6 1222.7 1232.8 1242.7 1213.1 1213.6 1203.1];
C2_LUT = [4264 72484 2781.7 83.598 4003.2 3429.2 3406.1 1874.8 3540.7 4881.8];
R0_LUT = [0.13898 0.19594 0.14919 0.12829 0.11056 0.10239 0.098289 0.099095 0.098839 0.079273];
R1_LUT = [0.15251 0.097901 0.0076771 0.0044429 0.013696 0.016048 0.0095527 0.017601 0.018307 0.0048358];
R2_LUT = [0.1486 1.8744e-05 0.056018 0.06936 0.061599 0.019188 0.0061243 0.061397 0.036596 0.0067722];
Em_LUT = [3.2046 3.4857 3.603 3.7293 3.7732 3.7852 3.8322 3.9335 4.0459 4.1542];

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
