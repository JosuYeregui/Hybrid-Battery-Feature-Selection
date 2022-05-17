function [parameters, functions] = ModelParameters(fileName)
sheet = 'Parameters';

%% GENERATE PARAMETER STRUCTURE TO LOAD ALL THE MODEL PARAMETERS FROM EXCEL SHEET
% Cell parameters have been obtained with the physico-chemical parameter
% obtention procedure.

[~,~,data]  = xlsread(fileName,sheet);

i = 1;

% Cell electrical properties
values(1,i) = data{10,2};
variables{1,i}=data{10,1};
i=i+1;
values(1,i) = data{11,2};
variables{1,i}=data{11,1};

% Operational parameters
for j = 1:10
values(1,j+i) = data{j+14,2};
variables{1,j+i}=data{j+14,1};
end
i = i+j;

% External dimensions
for j = 1:6
values(1,j+i) = data{j+31,2};
variables{1,j+i}=data{j+31,1};
end
i = i+j;

% Full battery thermal parameters
for j = 1:3
values(1,j+i) = data{j+40,2};
variables{1,j+i}=data{j+40,1};
end
i = i+j;

% Component composition and bulk properties 
%(thermodynamic, kinetic and transport properties)
values(1,i) = data{48,2};
variables{1,i}=data{48,1};
i=i+1;
values(1,i) = data{50,2};
variables{1,i}=data{50,1};
i=i+1;
values(1,i) = data{52,2};
variables{1,i}=data{52,1};
i=i+1;
values(1,i) = data{55,2};
variables{1,i}=data{55,1};

for j = 1:7
values(1,j+i) = data{j+58,2};
variables{1,j+i}=data{j+58,1};
end
i = i+j;

for j = 1:6
values(1,j+i) = data{j+68,2};
variables{1,j+i}=data{j+68,1};
end
i = i+j;

for j = 1:7
values(1,j+i) = data{j+76,2};
variables{1,j+i}=data{j+76,1};
end
i = i+j;

for j = 1:4
values(1,j+i) = data{j+86,2};
variables{1,j+i}=data{j+86,1};
end
i = i+j;

% OCV parameters
for j = 1:4
values(1,j+i) = data{j+91,2};
variables{1,j+i}=data{j+91,1};
end
i = i+j;

% Geometrical properties
for j = 1:7
values(1,j+i) = data{j+106,2};
variables{1,j+i}=data{j+106,1};
end
i = i+j;

% Solid and liquid volume fractions
values(1,i) = data{116,2};
variables{1,i}=data{116,1};
i=i+1;
values(1,i) = data{118,2};
variables{1,i}=data{118,1};
i=i+1;
values(1,i) = data{119,2};
variables{1,i}=data{119,1};
i=i+1;

% Bruggeman coefficients
values(1,i) = data{121,2};
variables{1,i}=data{121,1};
i=i+1;
values(1,i) = data{123,2};
variables{1,i}=data{123,1};
i=i+1;
values(1,i) = data{125,2};
variables{1,i}=data{125,1};

% Cell balancing
for j = 1:6
values(1,j+i) = data{j+134,2};
variables{1,j+i}=data{j+134,1};
end
i = i+j;

% Operational parameters: frequency model
for j = 1:4
values(1,j+i) = data{j+153,2};
variables{1,j+i}=data{j+153,1};
end
i = i+j;

% Extra forgoten parameters
values(1,i) = data{13,2};
variables{1,i}=data{13,1}; % C_nom
i=i+1;
values(1,i) = data{117,2};
variables{1,i}=data{117,1}; % eps_s_neg
i=i+1;
values(1,i) = data{115,2};
variables{1,i}=data{115,1}; % eps_s_pos
i=i+1;
values(1,i) = data{38,2};
variables{1,i}=data{38,1}; % A_cell
i=i+1;
values(1,i) = data{98,2};
variables{1,i}=data{98,1}; % cs_max_pos
i=i+1;
values(1,i) = data{99,2};
variables{1,i}=data{99,1}; %cs_max_neg

for j = 1:4 % SoL 0% and 100% SOC
values(1,j+i) = data{j+147,2};
variables{1,j+i}=data{j+147,1};
end
i = i+j;

parameters=struct('variables',variables','values',[]);

for k=1:length(parameters)
    parameters(k).values=values(k);
end

%% GENERATE MATRIX FILES FOR THE FUNCTIONS OF THE MODEL
% Change the OCV curves to take from experimental data
% load('OCV_LauraKokam.mat');
sheet2 = 'Functions';
[~,~,data2]  = xlsread(fileName,sheet2);

% Negative electrode
functions.Eeq_neg  = [data2(2:101,1), data2(2:101,2)];
functions.dEeqdT_neg  = [data2(2:101,4), data2(2:101,5)];
% functions.neg.soc_ref{1, 1} = [OCV_LauraKokam.SOC_neg_FC,OCV_LauraKokam.SOC_negative_FC];  % Normalization for Plett's model

% Positive electrode
functions.Eeq_pos  = [data2(2:101,7), data2(2:101,8)];
functions.dEeqdT_pos  = [data2(2:101,10), data2(2:101,11)];
% functions.pos.soc_ref{1, 1} = [flip(OCV_LauraKokam.SOC_pos_FC), OCV_LauraKokam.SOC_positive_FC];  % Normalization for Plett's model

% Electrolyte
functions.DL_int1  = [data2(2:4,13), data2(2:4,14)];
functions.sigmal_int1  = [data2(2:4,16), data2(2:4,17)];
functions.transpNm_int1  = [data2(2:8,19), data2(2:8,20)];

% Exchange current density
functions.i0_neg  = [data2(2:101,28), data2(2:101,29)];
functions.i0_pos  = [data2(2:101,22), data2(2:101,23)];
% functions.Ds_neg  = [data2(2:100,31), data2(2:100,32)];
% functions.Ds_pos  = [data2(2:101,25), data2(2:101,26)];
functions.Ds_neg  = [data2(2:23,31), data2(2:23,32)];
functions.Ds_pos  = [data2(2:11,25), data2(2:11,26)];
functions.Rfilm_neg  = [data2(2:100,34), data2(2:100,35)];
functions.Rfilm_pos  = [data2(2:101,37), data2(2:101,38)];

end