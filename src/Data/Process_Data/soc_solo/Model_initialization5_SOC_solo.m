%% Reference model: ssc_lithium_cell_2RC
clear all; clc;

%% Load Real experimental test information
% load RealTestData;
load('Perfil_ID.mat');

% Calculate capacity at different temperatures
% 25deg
Cbat_test_25 = abs(Perfil_ID(3).C_Ah(354865+232:1607770+2700,1));
Cbat_test_total_25 = sum(Cbat_test_25)/1000
% 45deg
Cbat_test_45 = abs(Perfil_ID(1).C_Ah(503592+170:1762410,1));
Cbat_test_total_45 = sum(Cbat_test_45)/1000
% 5deg
Cbat_test_05 = abs(Perfil_ID(5).C_Ah(353030:1608080,1));
Cbat_test_total_05 = sum(Cbat_test_05)/1000



% Time_test = unique(Time_test_raw);
% Time_test = (0:10^-4:Time_test_raw(end));
% Vbat_test = Vbat_test_raw;
% Vbat_test = interp1(Time_test_raw,Vbat_test_raw,Time_test);
% Vbat_test = interp1(Time_test_raw,Vbat_test_raw,Time_test,'nearest');

% T = 45 deg Perfil_ID
% Time_test = Perfil_ID(1).t_h(503592+170:1762410,1)-min(Perfil_ID(1).t_h(503592+170:1762410,1));
% Vbat_test= (Perfil_ID(1).V_V(503592+170:1762410,1));
% Ibat_test = Perfil_ID(1).I_A(503592+170:1762410,1);
% Tbat_test = Perfil_ID(1).T_deg(503592+170:1762410,1)+273.15;
% figure()
% plot(Time_test,Vbat_test)

% T = 45 deg Perfil_val
% Time_test = Perfil_ID(1).t_h(1:503592,1)-min(Perfil_ID(1).t_h(1:503592,1));
% Vbat_test= (Perfil_ID(1).V_V(1:503592,1));
% Ibat_test = Perfil_ID(1).I_A(1:503592,1);
% Tbat_test = Perfil_ID(1).T_deg(1:503592,1)+273.15;


% T = 25 deg
Time_test = Perfil_ID(3).t_h(354865+232:1607770+2700,1)-min(Perfil_ID(3).t_h(354865+232:1607770+2700,1));
Vbat_test = Perfil_ID(3).V_V(354865+232:1607770+2700,1);
Ibat_test = Perfil_ID(3).I_A(354865+232:1607770+2700,1);
Tbat_test = Perfil_ID(3).T_deg(354865+232:1607770+2700,1)+273.15;
figure()
plot(Time_test,Vbat_test)
figure()
plot(Time_test,Cbat_test)

% T = 5 deg
% Time_test = Perfil_ID(5).t_h(353030:1608080,1)-min(Perfil_ID(5).t_h(353030:1608080,1));
% Vbat_test = Perfil_ID(5).V_V(353030:1608080,1);
% Ibat_test = Perfil_ID(5).I_A(353030:1608080,1);
% Tbat_test = Perfil_ID(5).T_deg(353030:1608080,1)+273.15;

% Load OCV files
load('SOC_OCV.mat');

%% Load Characterization test information (Pulses test)
% load Time;
% load Voltage;
% load Ibat;
% load Tbat;

SIM_Ibat = Ibat_test;
SIM_Tbat = Tbat_test;
SIM_time = Time_test;
SIM_Vbat = Vbat_test;

%% Creating necessary vectors
% load Capacity_LUT;
% load SOC_LUT;
% load Temperature_LUT;
% load Em_LUT;

SOC_full = flip(SOC_OCVreal(:,1)*100);
SOC_LUT = linspace(SOC_full(1), SOC_full(end),10);
% Temperature_LUT = [5+273.15, 25+273.15, 45+273.15] ;
% Em_LUT = flip(SOC_OCVreal(:,[2:4]));
Em_full = flip(SOC_OCVreal(:,4));
Em_LUT = interp1(SOC_full,Em_full,SOC_LUT);

load C1_LUT;
load C2_LUT;
load R0_LUT;
load R1_LUT;
load R2_LUT;

% C1_LUT = [388.94;1270.9;1273;1272.3;1272.6;1271.6;1273.4;1271.1;1272.7;1271.7]
% C2_LUT = [3217.7;3548.2;3550.9;3549.8;3550.2;3549.6;3550.6;3549.1;3550;3549.7]
% R0_LUT = [0.0023592;0.0024054;0.0024061;0.0024051;0.002405;0.0024046;0.0024051;0.0024046;0.0024053;0.0024049]
% R1_LUT = [0.00076336;0.0011997;0.001201;0.0012005;0.0012004;0.0012004;0.0012004;0.0012005;0.0012005;0.0012005]
% R2_LUT = [0.0079508;0.0072945;0.0073047;0.0073008;0.0073006;0.007301;0.0073001;0.0073014;0.0073008;0.0073012]
% Em_LUT = [2.7023 3.4337 3.6227 3.7303 3.7739 3.7948 3.8501 3.947 4.0735 4.1654]

%% Initial and nominal capacities
% cell_area = 0.1019;
% cell_Cp_heat = 810.5328;
% cell_height = 0.2200;
% cell_mass = 1 ;
% cell_rho_Cp = 2040000;
% cell_thickness = 0.0084;
% cell_volume = 3.9732e-4;
% cell_width = 0.215;
% h_conv = 5;
% Qe_init = 0.8;

T_init = 273.15 + 45; % K
Rconv = 6.15;
Rcond = 0;
Cth = 25.95;
% Qnom = 1.25;
Qnom = 1.3359;
% Q0 = 1.25;
Q0 = 1.3359;