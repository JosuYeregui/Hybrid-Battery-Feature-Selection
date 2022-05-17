function [PulseResult] = sim_parpool_pulses(var2, parameters, model,s)

for v = 1:length(parameters);
    name = parameters(v).variables;
    model.param.set(name,var2(v)); 
end
       
model.study('std1').run;

PulseResult.SOCCell0 = var2(11);
PulseResult.T0 = var2(3);

% DEPENDENT VARIABLES
PulseResult.phis = mpheval(model,'phis','dataset', 'dset1');
PulseResult.phil = mpheval(model,'phil','dataset', 'dset1');
PulseResult.cl = mpheval(model,'cl','dataset', 'dset1');
PulseResult.cs_surface = mpheval(model,'liion.cs_surface','dataset', 'dset1');
PulseResult.T = mpheval(model,'T','dataset', 'dset1');
            
% CELL CHARACTERISTICS
PulseResult.time = mpheval(model,'t','edim','boundary','selection',1,'dataset', 'dset1');
PulseResult.current = mpheval(model,'liion.nIs','edim','boundary','selection',4,'dataset','dset1');
PulseResult.Ecell = mphglobal(model,'Ecell','dataset', 'dset1'); 
PulseResult.Epos = mphglobal(model,'Epos','dataset', 'dset1');
PulseResult.Eneg = mphglobal(model,'Eneg','dataset', 'dset1');
PulseResult.capa = mphglobal(model,'CAh_discharge','dataset', 'dset1');
PulseResult.enerden = mphglobal(model,'Em_discharge','dataset', 'dset1');
PulseResult.poden = mphglobal(model,'Pm_discharge','dataset', 'dset1');
PulseResult.Qtot = mphglobal(model,'Q_tot','dataset', 'dset1');
PulseResult.R0_estim = mpheval(model,'R','edim','boundary','selection',4,'dataset', 'dset1');
PulseResult.SOCneg_ave = mpheval(model,'socneg_ave','edim','boundary','selection',1,'dataset', 'dset1');
PulseResult.SOCpos_ave = mpheval(model,'socpos_ave','edim','boundary','selection',4,'dataset', 'dset1');
PulseResult.SOCneg_load = mpheval(model,'socneg_load','edim','boundary','selection',1,'dataset', 'dset1');
PulseResult.SOCpos_load = mpheval(model,'socpos_load','edim','boundary','selection',4,'dataset', 'dset1');
PulseResult.SOCCell = mpheval(model,'SOCCell','edim','boundary','selection',4,'dataset', 'dset1');       
end