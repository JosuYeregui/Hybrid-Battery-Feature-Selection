function [FreqResult] = sim_parpool_freq(var3, parameters, model,s)  

for v = 1:length(parameters);
    name = parameters(v).variables;
    model.param.set(name,var3(v)); 
end
       
model.study('std4').run;

FreqResult.SOCCell0 = var3(11);
FreqResult.T0 = var3(3);

% DEPENDENT VARIABLES
FreqResult.phis = mpheval(model,'phis2','dataset', 'dset7');
FreqResult.phil = mpheval(model,'phil2','dataset', 'dset7');
FreqResult.cl = mpheval(model,'cl2','dataset', 'dset7');
FreqResult.cs_surface = mpheval(model,'liion2.cs_surface','dataset', 'dset7');
            
% CELL CHARACTERISTICS
% FreqResult.time = mpheval(model,'t','edim','boundary','selection',1,'dataset', 'dset7');
% FreqResult.current = mpheval(model,'liion2.nIs','edim','boundary','selection',4,'dataset','dset7');
FreqResult.Ecell2 = mphglobal(model,'Ecell2','dataset', 'dset7'); 
% FreqResult.Epos2 = mphglobal(model,'Epos2','dataset', 'dset7');
% FreqResult.Eneg2 = mphglobal(model,'Eneg2','dataset', 'dset7');
% PulseResult.capa = mphglobal(model,'CAh_discharge','dataset', 'dset1');
% PulseResult.enerden = mphglobal(model,'Em_discharge','dataset', 'dset1');
% PulseResult.poden = mphglobal(model,'Pm_discharge','dataset', 'dset1');
% PulseResult.Qtot = mphglobal(model,'Q_tot','dataset', 'dset1');
% PulseResult.R0_estim = mpheval(model,'R','edim','boundary','selection',4,'dataset', 'dset1');
FreqResult.SOCneg_ave2 = mpheval(model,'socneg_ave2','edim','boundary','selection',1,'dataset', 'dset7');
FreqResult.SOCpos_ave2 = mpheval(model,'socpos_ave2','edim','boundary','selection',4,'dataset', 'dset7');
FreqResult.SOCneg_load2 = mpheval(model,'socneg_load2','edim','boundary','selection',1,'dataset', 'dset7');
FreqResult.SOCpos_load2 = mpheval(model,'socpos_load2','edim','boundary','selection',4,'dataset', 'dset7');
% PulseResult.SOCCell = mpheval(model,'SOCCell','edim','boundary','selection',4,'dataset', 'dset1'); 
FreqResult.Z_ground = mpheval(model,'Z_ground','edim','boundary','selection',4,'dataset', 'dset7');
FreqResult.Z_ref_NEG = mpheval(model,'Z_ref_NEG','edim','boundary','selection',4,'dataset', 'dset7');
FreqResult.Z_ref_POS = mpheval(model,'Z_ref_POS','edim','boundary','selection',4,'dataset', 'dset7');

end