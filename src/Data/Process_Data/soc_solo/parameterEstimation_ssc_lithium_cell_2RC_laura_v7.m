function [pOpt, Info] = parameterEstimation_ssc_lithium_cell_2RC_laura_v7(p)
%PARAMETERESTIMATION_SSC_LITHIUM_CELL_2RC_LAURA_V7
%
% Solve a parameter estimation problem for the ssc_lithium_cell_2RC_laura_v7 model.
%
% The function returns estimated parameter values, pOpt,
% and estimation termination information, Info.
%
% The input argument, p, defines the model parameters to estimate,
% if omitted the parameters specified in the function body are estimated.
%
% Modify the function to include or exclude new experiments, or
% to change the estimation options.
%
% Auto-generated by SPETOOL on 24-Mar-2022 16:02:31.
%

%% Open the model.
open_system('ssc_lithium_cell_2RC_laura_v7')

%% Specify Model Parameters to Estimate
%
if nargin < 1 || isempty(p)
    p = sdo.getParameterFromModel('ssc_lithium_cell_2RC_laura_v7',{'C1_LUT','C2_LUT','Em_LUT','R0_LUT','R1_LUT','R2_LUT','Rconv'});
    p(1).Minimum = getData('p1_Minimum');
    p(2).Minimum = getData('p2_Minimum');
    p(3).Minimum = getData('p3_Minimum');
    p(3).Maximum = getData('p3_Maximum');
    p(4).Minimum = getData('p4_Minimum');
    p(4).Maximum = [1 1 1 1 1 1 1 1 1 1];
    p(5).Minimum = getData('p5_Minimum');
    p(5).Maximum = [1 1 1 1 1 1 1 1 1 1];
    p(6).Minimum = getData('p6_Minimum');
    p(6).Maximum = [1 1 1 1 1 1 1 1 1 1];
    p(7).Minimum = 1e-06;
    p(7).Maximum = 100;
end

%% Define the Estimation Experiments
%

Exp = sdo.Experiment('ssc_lithium_cell_2RC_laura_v7');

%%
% Specify the experiment input data used to generate the output.
Exp_Sig_Input = Simulink.SimulationData.Signal;
Exp_Sig_Input.Values    = getData('Exp_Sig_Input_Value');
Exp_Sig_Input.BlockPath = 'ssc_lithium_cell_2RC_laura_v7/Input current';
Exp_Sig_Input.PortType  = 'outport';
Exp_Sig_Input.PortIndex = 1;
Exp_Sig_Input.Name      = 'Input current';
Exp.InputData = Exp_Sig_Input;

%%
% Specify the measured experiment output data.
Exp_Sig_Output_1 = Simulink.SimulationData.Signal;
Exp_Sig_Output_1.Values    = getData('Exp_Sig_Output_1_Value');
Exp_Sig_Output_1.BlockPath = 'ssc_lithium_cell_2RC_laura_v7/Electrical model';
Exp_Sig_Output_1.PortType  = 'outport';
Exp_Sig_Output_1.PortIndex = 1;
Exp_Sig_Output_1.Name      = 'ssc_lithium_cell_2RC_laura_v7/Electrical model:1';
Exp_Sig_Output_2 = Simulink.SimulationData.Signal;
Exp_Sig_Output_2.Values    = getData('Exp_Sig_Output_2_Value');
Exp_Sig_Output_2.BlockPath = 'ssc_lithium_cell_2RC_laura_v7/Thermal model';
Exp_Sig_Output_2.PortType  = 'outport';
Exp_Sig_Output_2.PortIndex = 1;
Exp_Sig_Output_2.Name      = 'ssc_lithium_cell_2RC_laura_v7/Thermal model:1';
Exp.OutputData = [Exp_Sig_Output_1; Exp_Sig_Output_2];

%%
% Create a model simulator from an experiment
Simulator = createSimulator(Exp);

%%
% Configure the simulator for fast restart.  Set place holder for
% OutputTimeValues, to be assigned in objective function.
Simulator = fastRestart(Simulator,'on');

%% Create Estimation Objective Function
%
% Create a function that is called at each optimization iteration
% to compute the estimation cost.
%
% Use an anonymous function with one argument that calls ssc_lithium_cell_2RC_laura_v7_optFcn.
optimfcn = @(P) ssc_lithium_cell_2RC_laura_v7_optFcn(P,Simulator,Exp);

%% Optimization Options
%
% Specify optimization options.
Options = sdo.OptimizeOptions;
Options.Method = 'lsqnonlin';
Options.OptimizedModel = Simulator;
Options.UseParallel = 1;
Options.ParallelFileDependencies = getData('Options_ParallelFileDependencies');
Options.ParallelFileDependencies{end+1} = mfilename('fullpath');

%% Estimate the Parameters
%
% Call sdo.optimize with the estimation objective function handle,
% parameters to estimate, and options.
[pOpt,Info] = sdo.optimize(optimfcn,p,Options);

%%
% Restore the simulator fast restart settings
Simulator = fastRestart(Simulator,'off');

%%
% Update the experiments with the estimated parameter values.
Exp = setEstimatedValues(Exp,pOpt);

%% Update Model
%
% Update the model with the optimized parameter values.
sdo.setValueInModel('ssc_lithium_cell_2RC_laura_v7',pOpt);
end

function Vals = ssc_lithium_cell_2RC_laura_v7_optFcn(P,Simulator,Exp)
%SSC_LITHIUM_CELL_2RC_LAURA_V7_OPTFCN
%
% Function called at each iteration of the estimation problem.
%
% The function is called with a set of parameter values, P, and returns
% the estimation cost, Vals, to the optimization solver.
%
% See the sdoExampleCostFunction function and sdo.optimize for a more
% detailed description of the function signature.
%

%%
% Define a signal tracking requirement to compute how well the model
% output matches the experiment data.
r = sdo.requirements.SignalTracking(...
    'Method', 'Residuals');
%%
% Update the experiment(s) with the estimated parameter values.
Exp = setEstimatedValues(Exp,P);

%%
% Simulate the model and compare model outputs with measured experiment
% data.

F_r = [];
Simulator = createSimulator(Exp,Simulator);
Simulator = sim(Simulator);

SimLog = find(Simulator.LoggedData,get_param('ssc_lithium_cell_2RC_laura_v7','SignalLoggingName'));
for ctSig=1:numel(Exp.OutputData)
    Sig = find(SimLog,Exp.OutputData(ctSig).Name);

    Error = evalRequirement(r,Sig.Values,Exp.OutputData(ctSig).Values);
    F_r = [F_r; Error(:)];
end

%% Return Values.
%
% Return the evaluated estimation cost in a structure to the
% optimization solver.
Vals.F = F_r;
end

function Data = getData(DataID)
%GETDATA
%
% Helper function to store data used by parameterEstimation_ssc_lithium_cell_2RC_laura_v7.
%
% The input, DataID, specifies the name of the data to retrieve. The output,
% Data, contains the requested data.
%

SaveData = load('parameterEstimation_ssc_lithium_cell_2RC_laura_v7_Data');
Data = SaveData.Data.(DataID);
end