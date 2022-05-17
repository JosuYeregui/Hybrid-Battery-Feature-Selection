function [OneResult] = sim_parpool_galvCH_DCH(var, parameters, model,c)

for v = 1:length(parameters);
    name = parameters(v).variables;
    model.param.set(name,var(v)); 
end

   % Change only operational parameters to change T and Crate (NOT ALL THE EXCEL PARAMETERS)
%    model.param.set('T0',var(3));
%    model.param.set('C_rate_charge_set',var(9));
%    model.param.set('C_rate_discharge_set',var(10));
%    model.param.set('SOCCell0',var(11));
%    model.param.set('C_rate',var(12));
%    model.param.set('SpecifyTime',var(4));
       
i = 1;

%         model.param.set('T0',Temp(c));
%         for c = 1:13 % hasta 13 performance at different Crates (on charge and discharge)
           if var(12)> 0 % stop conditions of the charge process
%              model.param.set('C_rate_charge_set',Crate(c));
%              model.param.set('SOCCell0',0);
            model.study('std3').run;
%             model.sol('sol86').runAll;


%             enerden = mphglobal(model,'Em_charge','dataset', 'dset24');
%             % enerden = mphglobal(model,'Em_out');
%            enerden = enerden(end);
%             variables{i}='enerden';
%             i = i+1;

%             poden = mphglobal(model,'Pm_charge','dataset', 'dset24');
%             % poden = mphglobal(model,'Pm_out');
%             poden = poden(end);
%             variables{i}='poden';
%             i = i+1;

            OneResult.Crate = var(12);
            OneResult.T0 = var(3);
%                  % DEPENDENT VARIABLES
                 OneResult.phis = mpheval(model,'phis','dataset', 'dset24');
                 OneResult.phil = mpheval(model,'phil','dataset', 'dset24');
                 OneResult.cl = mpheval(model,'cl','dataset', 'dset24');
                 OneResult.cs_surface = mpheval(model,'liion.cs_surface','dataset', 'dset24');
                 OneResult.T = mpheval(model,'T','dataset', 'dset24');
%                  % eta and jn?
%                  % CELL CHARACTERISTICS
                 OneResult.time = mpheval(model,'t','edim','boundary','selection',1,'dataset', 'dset24');
                 OneResult.current = mpheval(model,'i_app_charge_norm','edim','boundary','selection',4,'dataset', 'dset24');
                 OneResult.Ecell = mphglobal(model,'Ecell','dataset', 'dset24');
                 OneResult.Epos = mphglobal(model,'Epos','dataset', 'dset24');
                 OneResult.Eneg = mphglobal(model,'Eneg','dataset', 'dset24');
                 OneResult.capa = mphglobal(model,'CAh_charge','dataset', 'dset24');
                 OneResult.enerden = mphglobal(model,'Em_charge','dataset', 'dset24');
                 OneResult.poden = mphglobal(model,'Pm_charge','dataset', 'dset24');
                 OneResult.Qtot = mphglobal(model,'Q_tot','dataset', 'dset24');
                 OneResult.R0_estim = mpheval(model,'R','edim','boundary','selection',4,'dataset', 'dset24');
                 OneResult.SOCneg_ave = mpheval(model,'socneg_ave','edim','boundary','selection',1,'dataset', 'dset24');
                 OneResult.SOCpos_ave = mpheval(model,'socpos_ave','edim','boundary','selection',4,'dataset', 'dset24');
                 OneResult.SOCneg_load = mpheval(model,'socneg_load','edim','boundary','selection',1,'dataset', 'dset24');
                 OneResult.SOCpos_load = mpheval(model,'socpos_load','edim','boundary','selection',4,'dataset', 'dset24');
                 OneResult.SOCCell = mpheval(model,'SOCCell','edim','boundary','selection',4,'dataset', 'dset24');
                    
                 else % stop conditions of the discharge process
% %                  model.param.set('C_rate_discharge_set',Crate(c)*(-1));
% %                  model.param.set('SOCCell0',1);
                  model.study('std2').run;
                  
%                  enerden = mphglobal(model,'Em_discharge','dataset', 'dset22');
%                 % enerden = mphglobal(model,'Em_out');
%                 enerden = enerden(end);
%                 variables{i}='enerden';
%                 i = i+1;
% 
%                 poden = mphglobal(model,'Pm_discharge','dataset', 'dset22');
%                 % poden = mphglobal(model,'Pm_out');
%                 poden = poden(end);
%                 variables{i}='poden';
%                 i = i+1;
%                  model.sol('sol84').runAll;
% %                  model.sol('sol84').feature('t1').feature('st1').setIndex('eventstopActive', true, 1);
%                  
                 OneResult.Crate = var(12);
                 OneResult.T0 = var(3);
%                  % DEPENDENT VARIABLES
                 OneResult.phis = mpheval(model,'phis','dataset', 'dset22');
                 OneResult.phil = mpheval(model,'phil','dataset', 'dset22');
                 OneResult.cl = mpheval(model,'cl','dataset', 'dset22');
                 OneResult.cs_surface = mpheval(model,'liion.cs_surface','dataset', 'dset22');
                 OneResult.T = mpheval(model,'T','dataset', 'dset22');
%                  % eta and jn?
%                  % CELL CHARACTERISTICS
                 OneResult.time = mpheval(model,'t','edim','boundary','selection',1,'dataset', 'dset22');
                 OneResult.current = mpheval(model,'i_app_discharge_norm','edim','boundary','selection',4,'dataset', 'dset22');
                 OneResult.Ecell = mphglobal(model,'Ecell','dataset', 'dset22'); 
                 OneResult.Epos = mphglobal(model,'Epos','dataset', 'dset22');
                 OneResult.Eneg = mphglobal(model,'Eneg','dataset', 'dset22');
                 OneResult.capa = mphglobal(model,'CAh_discharge','dataset', 'dset22');
                 OneResult.enerden = mphglobal(model,'Em_discharge','dataset', 'dset22');
                 OneResult.poden = mphglobal(model,'Pm_discharge','dataset', 'dset22');
                 OneResult.Qtot = mphglobal(model,'Q_tot','dataset', 'dset22');
                 OneResult.R0_estim = mpheval(model,'R','edim','boundary','selection',4,'dataset', 'dset22');
                 OneResult.SOCneg_ave = mpheval(model,'socneg_ave','edim','boundary','selection',1,'dataset', 'dset22');
                 OneResult.SOCpos_ave = mpheval(model,'socpos_ave','edim','boundary','selection',4,'dataset', 'dset22');
                 OneResult.SOCneg_load = mpheval(model,'socneg_load','edim','boundary','selection',1,'dataset', 'dset22');
                 OneResult.SOCpos_load = mpheval(model,'socpos_load','edim','boundary','selection',4,'dataset', 'dset22');
                 OneResult.SOCCell = mpheval(model,'SOCCell','edim','boundary','selection',4,'dataset', 'dset22');
           end
         
           
%            Results = [enerden(end), poden(end)] ;
%            Names = {'Em', 'Pm'};
            
%         Galvanostatic(c).Crates = Crates; 
        
end