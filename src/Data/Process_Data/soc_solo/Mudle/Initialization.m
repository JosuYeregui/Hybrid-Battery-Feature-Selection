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
    R0_LUT = [0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024];
    R1_LUT = [0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012];
    R2_LUT = [0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073];
    C1_LUT = [1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3];
    C2_LUT = [3.55e3 3.55e3 3.55e3 3.55e3 3.55e3 3.55e3 3.55e3 3.55e3 3.55e3 3.55e3];
    
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
    R0_LUT = [0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024];
    R1_LUT = [0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012];
    R2_LUT = [0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073];
    C1_LUT = [1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3];
    C2_LUT = [3.55e3 3.55e3 3.55e3 3.55e3 3.55e3 3.55e3 3.55e3 3.55e3 3.55e3 3.55e3];
    
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
    R0_LUT = [0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024 0.0024];
    R1_LUT = [0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012 0.0012];
    R2_LUT = [0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073 0.0073];
    C1_LUT = [1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3 1.2723e3];
    C2_LUT = [3.55e3 3.55e3 3.55e3 3.55e3 3.55e3 3.55e3 3.55e3 3.55e3 3.55e3 3.55e3];
    
    % Guardar valores de los parámetros en un lugar común para todas las Temperaturas
    Em_LUT_T(3,:) = Em_LUT;
    R0_LUT_T(3,:) = R0_LUT;
    C1_LUT_T(3,:) = C1_LUT;
    C2_LUT_T(3,:) = C2_LUT;
    R1_LUT_T(3,:) = R1_LUT;
    R2_LUT_T(3,:) = R2_LUT;
end
