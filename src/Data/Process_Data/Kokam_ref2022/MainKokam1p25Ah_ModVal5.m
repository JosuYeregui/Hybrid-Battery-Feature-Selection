%% SEQUENCED ANALYSIS FOR EXPERIMENTAL-NUMERICAL COMPARISON
% Performed steps:
% SIMULATION 1. Galvanostatic charge and discharge
% SIMULATION 2. Pulses
% SIMULATION 3. Frequency domain analysis

%% INITIALIZATION
close all; clear all; clc

%% LOAD PARAMETERS
[parameters, functions] = ModelParameters8('CELL_LauraKokam_parameters15z');

ResultFileName = 'CELL_LauraKokam_parameters15y_modelV18_15z2_22032020_v2.mat'; % Specify the name of the .m file of the results

%% OPEN 20 WORKERS + LOAD THE MODEL (PARFOR)
comsolPort=[3036 3037 3038 3039 3040 3041 3042 3043 3044 3045 3046 3047 3048 3049 3050 3051 3052 3053 3054 3055];
parfor i=1:length(comsolPort)    
comsolPort=[3036 3037 3038 3039 3040 3041 3042 3043 3044 3045 3046 3047 3048 3049 3050 3051 3052 3053 3054 3055];

t = getCurrentTask();
labit=t.ID;       

cd('C:\Program Files\COMSOL\COMSOL54\Multiphysics_copy1\bin\win64');
system( ['comsolmphserver.exe -np 1 -port ' num2str(comsolPort(labit)) ' &'] );
% pause(6)   
cd(('C:\Program Files\COMSOL\COMSOL54\Multiphysics_copy1\mli'))
mphstart(comsolPort(labit))
import('com.comsol.model.*');
import('com.comsol.model.util.*');
import ('com.comsol.model.util.disconnect.*');
% ModelUtil.showProgress(true);
cd('F:\Laura\KOKAM1-25Ah')
model = mphload('li_battery_root_v18_parameters15z3.mph',num2str(comsolPort(labit)));
end

%% SIMULATION 1: Galvanostatic charge-discharge 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
% Parameter changes: numero_parametros x numero_simulaciones % 3T x 13c = 39 simulations
% Temp = [273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45]; % cell initial temperature (K)
% Crate = [0.1 0.2 0.5 1 2 -0.1 -0.2 -0.5 -1 -2 -5 -10 -15 0.1 0.2 0.5 1 2 -0.1 -0.2 -0.5 -1 -2 -5 -10 -15 0.1 0.2 0.5 1 2 -0.1 -0.2 -0.5 -1 -2 -5 -10 -15];
% SOCCell0 = [0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 1 1 1 1 1 1 1 1];

Temp = [273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+25 273.15+25]; % cell initial temperature (K)
Crate = [0.1 0.2 0.5 1 2 -0.1 -0.2 -0.5 -1 -2 -5 0.1 0.2 0.5 1 2 -0.1 -0.2 -0.5 -1 -2 -5 0.1 0.2 0.5 1 2 -0.1 -0.2 -0.5 -1 -2 -5 0.033 -0.033];
SOCCell0 = [0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 1 1 1 1 1 1 0 1];

% Load parameters
for i= 1:length(parameters)
  value2 = parameters(i).values;
    for j=1:length(Temp)
    vector(j) = value2;
    end
  Param(i,:) = [vector];
end
Param(3,:) = Temp; % Change T0
Param(9,:) = abs(Crate); % Change C_rate_charge_set
Param(10,:) = abs(Crate); % Change C_rate_discharge_set
Param(11,:) = SOCCell0; % Change SOCCell0
Param(12,:) = Crate; % Change Crate (with sign)
Param(4,:) = 0; % SpecifyTime, activate or desactivate this function (= 0 --> full charge or discharge; = 1 --> pulses)
        
% Run simulation (change T and Crate)
parfor c = 1:length(Temp) % 3T x 13c = 33 simulations
   model = openserver;
   
        % Load functions to the model
        % Electrolyte diffusion coefficient
        model.func('int1').label('DL_int1');
        for x = 1:length(functions.DL_int1(:,1))
        model.func('int1').setIndex('table', functions.DL_int1{x,1}, x-1, 0);
        model.func('int1').setIndex('table', functions.DL_int1{x,2}, x-1, 1);
        end
        % Electrolyte conductivity
        model.func('int2').label('sigmal_int1');
        for x = 1:length(functions.sigmal_int1(:,1))
        model.func('int2').setIndex('table', functions.sigmal_int1{x,1}, x-1, 0);
        model.func('int2').setIndex('table', functions.sigmal_int1{x,2}, x-1, 1);
        end
        % Transport number
        model.func('int3').label('transpNm_int1');
        for x = 1:length(functions.transpNm_int1(:,1))
        model.func('int3').setIndex('table', functions.transpNm_int1{x,1}, x-1, 0);
        model.func('int3').setIndex('table', functions.transpNm_int1{x,2}, x-1, 1);
        end
        % Eeq_neg
        model.func('int5').label('Eeq_neg');
        for x = 1:length(functions.Eeq_neg(:,1))
        model.func('int5').setIndex('table', functions.Eeq_neg{x,1}, x-1, 0);
        model.func('int5').setIndex('table', functions.Eeq_neg{x,2}, x-1, 1);
        end
        % dEeqdT_neg
        model.func('int6').label('dEeqdT_neg');
        for x = 1:length(functions.dEeqdT_neg(:,1))
        model.func('int6').setIndex('table', functions.dEeqdT_neg{x,1}, x-1, 0);
        model.func('int6').setIndex('table', functions.dEeqdT_neg{x,2}, x-1, 1);
        end
        % Eeq_pos
        model.func('int7').label('Eeq_pos');
        for x = 1:length(functions.Eeq_pos(:,1))
        model.func('int7').setIndex('table', functions.Eeq_pos{x,1}, x-1, 0);
        model.func('int7').setIndex('table', functions.Eeq_pos{x,2}, x-1, 1);
        end
        % dEeqdT_pos
        model.func('int8').label('dEeqdT_pos');
        for x = 1:length(functions.dEeqdT_pos(:,1))
        model.func('int8').setIndex('table', functions.dEeqdT_pos{x,1}, x-1, 0);
        model.func('int8').setIndex('table', functions.dEeqdT_pos{x,2}, x-1, 1);
        end
%         i0_pos
        model.func('int9').label('i0_pos');
        for x = 1:length(functions.i0_pos(:,1))
        model.func('int9').setIndex('table', functions.i0_pos{x,1}, x-1, 0);
        model.func('int9').setIndex('table', functions.i0_pos{x,2}, x-1, 1);
        end
        % Ds_pos
        model.func('int10').label('Ds_pos');
        for x = 1:length(functions.Ds_pos(:,1))
        model.func('int10').setIndex('table', functions.Ds_pos{x,1}, x-1, 0);
        model.func('int10').setIndex('table', functions.Ds_pos{x,2}, x-1, 1);
        end
        % i0_neg
        model.func('int11').label('i0_neg');
        for x = 1:length(functions.i0_neg(:,1))
        model.func('int11').setIndex('table', functions.i0_neg{x,1}, x-1, 0);
        model.func('int11').setIndex('table', functions.i0_neg{x,2}, x-1, 1);
        end
        % Ds_neg
        model.func('int12').label('Ds_neg');
        for x = 1:length(functions.Ds_neg(:,1))
        model.func('int12').setIndex('table', functions.Ds_neg{x,1}, x-1, 0);
        model.func('int12').setIndex('table', functions.Ds_neg{x,2}, x-1, 1);
        end
        % Rfilm_neg
        model.func('int13').label('Rfilm_neg');
        for x = 1:length(functions.Rfilm_neg(:,1))
        model.func('int13').setIndex('table', functions.Rfilm_neg{x,1}, x-1, 0);
        model.func('int13').setIndex('table', functions.Rfilm_neg{x,2}, x-1, 1);
        end
        % Rfilm_pos
        model.func('int14').label('Rfilm_pos');
        for x = 1:length(functions.Rfilm_pos(:,1))
        model.func('int14').setIndex('table', functions.Rfilm_pos{x,1}, x-1, 0);
        model.func('int14').setIndex('table', functions.Rfilm_pos{x,2}, x-1, 1);
        end
        
   var = Param(:,c);
   [OneResult] = sim_parpool_galvCH_DCH(var, parameters, model,c)
   Results(c,:) =[OneResult];
end

Simulation.Galvanostatic = Results;
save(ResultFileName,'Simulation');
toc

%% SIMULATION 2: Pulses (change T and SoC) "HPPC protocol"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic 
% Parameter changes
Temp_2 = [273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45]; % cell initial temperature (K)
SOCCell0_2 = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];

% Load parameters
for i=1:length(parameters)
  value3 = parameters(i).values;
    for j=1:length(Temp_2)
    vector2(j) = value3;
    end
  Param2(i,:) = [vector2];
end
Param2(3,:) = Temp_2; % Change T0
Param2(9,:) = 0.75; % Change C_rate_charge_set
Param2(10,:) = 1; % Change C_rate_discharge_set
Param2(11,:) = SOCCell0_2; % Change SOCCell0
Param2(12,:) = 1; % Change Crate (with sign)
Param2(4,:) = 1; % SpecifyTime
Param2(7,:) = 10; % t_charge_set, charge time (s)
Param2(8,:) = 10; % t_discharge_set, discharge time (s)

% Run simulation
parfor s = 1:length(Temp_2) % performance at different SoCs and T
    model = openserver;
%           Load functions to the model
        % Electrolyte diffusion coefficient
        model.func('int1').label('DL_int1');
        for x = 1:length(functions.DL_int1(:,1))
        model.func('int1').setIndex('table', functions.DL_int1{x,1}, x-1, 0);
        model.func('int1').setIndex('table', functions.DL_int1{x,2}, x-1, 1);
        end
        % Electrolyte conductivity
        model.func('int2').label('sigmal_int1');
        for x = 1:length(functions.sigmal_int1(:,1))
        model.func('int2').setIndex('table', functions.sigmal_int1{x,1}, x-1, 0);
        model.func('int2').setIndex('table', functions.sigmal_int1{x,2}, x-1, 1);
        end
        % Transport number
        model.func('int3').label('transpNm_int1');
        for x = 1:length(functions.transpNm_int1(:,1))
        model.func('int3').setIndex('table', functions.transpNm_int1{x,1}, x-1, 0);
        model.func('int3').setIndex('table', functions.transpNm_int1{x,2}, x-1, 1);
        end
        % Eeq_neg
        model.func('int5').label('Eeq_neg');
        for x = 1:length(functions.Eeq_neg(:,1))
        model.func('int5').setIndex('table', functions.Eeq_neg{x,1}, x-1, 0);
        model.func('int5').setIndex('table', functions.Eeq_neg{x,2}, x-1, 1);
        end
        % dEeqdT_neg
        model.func('int6').label('dEeqdT_neg');
        for x = 1:length(functions.dEeqdT_neg(:,1))
        model.func('int6').setIndex('table', functions.dEeqdT_neg{x,1}, x-1, 0);
        model.func('int6').setIndex('table', functions.dEeqdT_neg{x,2}, x-1, 1);
        end
        % Eeq_pos
        model.func('int7').label('Eeq_pos');
        for x = 1:length(functions.Eeq_pos(:,1))
        model.func('int7').setIndex('table', functions.Eeq_pos{x,1}, x-1, 0);
        model.func('int7').setIndex('table', functions.Eeq_pos{x,2}, x-1, 1);
        end
        % dEeqdT_pos
        model.func('int8').label('dEeqdT_pos');
        for x = 1:length(functions.dEeqdT_pos(:,1))
        model.func('int8').setIndex('table', functions.dEeqdT_pos{x,1}, x-1, 0);
        model.func('int8').setIndex('table', functions.dEeqdT_pos{x,2}, x-1, 1);
        end
        % i0_pos
        model.func('int9').label('i0_pos');
        for x = 1:length(functions.i0_pos(:,1))
        model.func('int9').setIndex('table', functions.i0_pos{x,1}, x-1, 0);
        model.func('int9').setIndex('table', functions.i0_pos{x,2}, x-1, 1);
        end
        % Ds_pos
        model.func('int10').label('Ds_pos');
        for x = 1:length(functions.Ds_pos(:,1))
        model.func('int10').setIndex('table', functions.Ds_pos{x,1}, x-1, 0);
        model.func('int10').setIndex('table', functions.Ds_pos{x,2}, x-1, 1);
        end
        % i0_neg
        model.func('int11').label('i0_neg');
        for x = 1:length(functions.i0_neg(:,1))
        model.func('int11').setIndex('table', functions.i0_neg{x,1}, x-1, 0);
        model.func('int11').setIndex('table', functions.i0_neg{x,2}, x-1, 1);
        end
        % Ds_neg
        model.func('int12').label('Ds_neg');
        for x = 1:length(functions.Ds_neg(:,1))
        model.func('int12').setIndex('table', functions.Ds_neg{x,1}, x-1, 0);
        model.func('int12').setIndex('table', functions.Ds_neg{x,2}, x-1, 1);
        end
                % Rfilm_neg
        model.func('int13').label('Rfilm_neg');
        for x = 1:length(functions.Rfilm_neg(:,1))
        model.func('int13').setIndex('table', functions.Rfilm_neg{x,1}, x-1, 0);
        model.func('int13').setIndex('table', functions.Rfilm_neg{x,2}, x-1, 1);
        end
        % Rfilm_pos
        model.func('int14').label('Rfilm_pos');
        for x = 1:length(functions.Rfilm_pos(:,1))
        model.func('int14').setIndex('table', functions.Rfilm_pos{x,1}, x-1, 0);
        model.func('int14').setIndex('table', functions.Rfilm_pos{x,2}, x-1, 1);
        end
        
var2 = Param2(:,s);
[PulseResult] = sim_parpool_pulses(var2, parameters, model,s)    
Pulses(s,:) =PulseResult;  
%     catch
% s
%     end
end
% closeserver;    
Simulation.Pulses = Pulses;
save(ResultFileName,'Simulation');
toc    

%% Frequency domain. (change T, SoC)
tic
% Parameter changes
Temp_3 = [273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+5 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+25 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45 273.15+45]; % cell initial temperature (K)
SOCCell0_3 = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];

% Load parameters
for i=1:length(parameters)
  value4 = parameters(i).values;
    for j=1:length(Temp_3)
    vector3(j) = value4;
    end
  Param3(i,:) = [vector3];
end
Param3(3,:) = Temp_3; % Change T0
Param3(11,:) = SOCCell0_3; % Change SOCCell0
Param3(4,:) = 0; % SpecifyTime
% Param3(71,:) = 0.01; % E_pert
% Param3(72,:) = 10000; % Max_freq
% Param3(73,:) = 0.001; % Min_freq

% Run simulation
parfor s = 1:length(Temp_3) % performance at different SoCs and T
    model = openserver;
% Load functions to the model
        % Electrolyte diffusion coefficient
        model.func('int1').label('DL_int1');
        for x = 1:length(functions.DL_int1(:,1))
        model.func('int1').setIndex('table', functions.DL_int1{x,1}, x-1, 0);
        model.func('int1').setIndex('table', functions.DL_int1{x,2}, x-1, 1);
        end
        % Electrolyte conductivity
        model.func('int2').label('sigmal_int1');
        for x = 1:length(functions.sigmal_int1(:,1))
        model.func('int2').setIndex('table', functions.sigmal_int1{x,1}, x-1, 0);
        model.func('int2').setIndex('table', functions.sigmal_int1{x,2}, x-1, 1);
        end
        % Transport number
        model.func('int3').label('transpNm_int1');
        for x = 1:length(functions.transpNm_int1(:,1))
        model.func('int3').setIndex('table', functions.transpNm_int1{x,1}, x-1, 0);
        model.func('int3').setIndex('table', functions.transpNm_int1{x,2}, x-1, 1);
        end
        % Eeq_neg
        model.func('int5').label('Eeq_neg');
        for x = 1:length(functions.Eeq_neg(:,1))
        model.func('int5').setIndex('table', functions.Eeq_neg{x,1}, x-1, 0);
        model.func('int5').setIndex('table', functions.Eeq_neg{x,2}, x-1, 1);
        end
        % dEeqdT_neg
        model.func('int6').label('dEeqdT_neg');
        for x = 1:length(functions.dEeqdT_neg(:,1))
        model.func('int6').setIndex('table', functions.dEeqdT_neg{x,1}, x-1, 0);
        model.func('int6').setIndex('table', functions.dEeqdT_neg{x,2}, x-1, 1);
        end
        % Eeq_pos
        model.func('int7').label('Eeq_pos');
        for x = 1:length(functions.Eeq_pos(:,1))
        model.func('int7').setIndex('table', functions.Eeq_pos{x,1}, x-1, 0);
        model.func('int7').setIndex('table', functions.Eeq_pos{x,2}, x-1, 1);
        end
        % dEeqdT_pos
        model.func('int8').label('dEeqdT_pos');
        for x = 1:length(functions.dEeqdT_pos(:,1))
        model.func('int8').setIndex('table', functions.dEeqdT_pos{x,1}, x-1, 0);
        model.func('int8').setIndex('table', functions.dEeqdT_pos{x,2}, x-1, 1);
        end
        % i0_pos
        model.func('int9').label('i0_pos');
        for x = 1:length(functions.i0_pos(:,1))
        model.func('int9').setIndex('table', functions.i0_pos{x,1}, x-1, 0);
        model.func('int9').setIndex('table', functions.i0_pos{x,2}, x-1, 1);
        end
        % Ds_pos
        model.func('int10').label('Ds_pos');
        for x = 1:length(functions.Ds_pos(:,1))
        model.func('int10').setIndex('table', functions.Ds_pos{x,1}, x-1, 0);
        model.func('int10').setIndex('table', functions.Ds_pos{x,2}, x-1, 1);
        end
        % i0_neg
        model.func('int11').label('i0_neg');
        for x = 1:length(functions.i0_neg(:,1))
        model.func('int11').setIndex('table', functions.i0_neg{x,1}, x-1, 0);
        model.func('int11').setIndex('table', functions.i0_neg{x,2}, x-1, 1);
        end
        % Ds_neg
        model.func('int12').label('Ds_neg');
        for x = 1:length(functions.Ds_neg(:,1))
        model.func('int12').setIndex('table', functions.Ds_neg{x,1}, x-1, 0);
        model.func('int12').setIndex('table', functions.Ds_neg{x,2}, x-1, 1);
        end
        % Rfilm_neg
        model.func('int13').label('Rfilm_neg');
        for x = 1:length(functions.Rfilm_neg(:,1))
        model.func('int13').setIndex('table', functions.Rfilm_neg{x,1}, x-1, 0);
        model.func('int13').setIndex('table', functions.Rfilm_neg{x,2}, x-1, 1);
        end
        % Rfilm_pos
        model.func('int14').label('Rfilm_pos');
        for x = 1:length(functions.Rfilm_pos(:,1))
        model.func('int14').setIndex('table', functions.Rfilm_pos{x,1}, x-1, 0);
        model.func('int14').setIndex('table', functions.Rfilm_pos{x,2}, x-1, 1);
        end
        
var3 = Param3(:,s);
[FreqResult] = sim_parpool_freq(var3, parameters, model,s)    
Freq(s,:) =FreqResult;  
end

Simulation.Frequency = Freq;
save(ResultFileName,'Simulation');
toc
