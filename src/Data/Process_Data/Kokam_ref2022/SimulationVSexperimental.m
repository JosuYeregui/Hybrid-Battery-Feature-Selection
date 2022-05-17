% Simulation vs experimental comparison

%% SIMULATION 1 GRAPHS VS EXPERIMENTAL (galvanostatic charge-discharges)
% Load experimental tests
load('test_info.mat');
load('CELL_LauraKokam_parameters15y_modelV18_15z2_22032020_v2.mat');

%% Ragone plot (Em/Pm) FIG 8.A: discharge and all T
% T1, T2, T3
figure()
% Simulation
for T = 6:13
    plot(Simulation.Galvanostatic(T).enerden(end,1), Simulation.Galvanostatic(T).poden(end,1),'Marker','o','Color','r')
    hold on
end
for T = 19:26
    plot(Simulation.Galvanostatic(T).enerden(end,1), Simulation.Galvanostatic(T).poden(end,1),'Marker','o','Color','b')
    hold on
end
for T = 32:39
    plot(Simulation.Galvanostatic(T).enerden(end,1), Simulation.Galvanostatic(T).poden(end,1),'Marker','o','Color','k')
    hold on
end

% Experimental discharge
for i =1:6 
    for j = 1:7
% specific energy [Wh/kg]
% specific power [W/kg]
scatter(test_info(i).test.Specific_Energy(1,j),test_info(i).test.Specific_Power(1,j))
hold on
    end
end
title('Ragone plot')
legend('273.15K', '298.15K', '318.15K')
xlabel('Energy density (W/kg)')
ylabel('Power density (Wh/kg)')
grid on


%% Voltage vs capacity plot (discharge)
% T1 = 278.15 K (5 ºC)
figure()
% Simulation
for T = 6:13
    plot(Simulation.Galvanostatic(T).capa, Simulation.Galvanostatic(T).Ecell)
    hold on
end
% Experimental
for i = 5:6
    for j = 1:5
        plot(flip(test_info(i).test.For_sim.Crate(j).capacity),test_info(i).test.For_sim.Crate(j).voltage)
        hold on
    end
end

title('Galvanostatic discharges at 278.15 K')
legend('0.1C num','0.2C num','0.5C num','1C num','2C num','5C num','10C num','15C num','0.1C exp','0.2C exp','0.5C exp','1C exp','2C exp','5C exp','10C exp')
xlabel('Capacity (Ah)')
ylabel('Voltage (V)')
grid on

figure()
% Simulation
for T = 6:13
    plot(Simulation.Galvanostatic(T).capa, Simulation.Galvanostatic(T).T.d1(:,1))
    hold on
end
% Experimental
for i = 5:6
    for j = 1:5
        plot(flip(test_info(i).test.For_sim.Crate(j).capacity),test_info(i).test.For_sim.Crate(j).temperature+273.15,'marker','x')
        hold on
    end
end
title('Galvanostatic discharges at 278.15 K and temperature dependence')
legend('0.1C num','0.2C num','0.5C num','1C num','2C num','5C num','10C num','15C num','0.1C exp','0.2C exp','0.5C exp','1C exp','2C exp','5C exp','10C exp')
xlabel('Capacity (Ah)')
ylabel('Temperature (K)')
grid on

%% T2 = 298.15 K (25 ºC) FIG 8.B discharge
figure()
% Simulation
for T = 19:26
    plot(Simulation.Galvanostatic(T).capa, Simulation.Galvanostatic(T).Ecell)
    hold on
end
% Experimental
for i = 3:4
    for j = 1:6
        plot(flip(test_info(i).test.For_sim.Crate(j).capacity),test_info(i).test.For_sim.Crate(j).voltage)
        hold on
    end
end
title('Galvanostatic discharges at 298.15 K')
legend('0.1C num','0.2C num','0.5C num','1C num','2C num','5C num','10C num','15C num','0.1C exp','0.2C exp','0.5C exp','1C exp','2C exp','5C exp','10C exp')
xlabel('Capacity (Ah)')
ylabel('Voltage (V)')
grid on

% Temperature behaviour 25 ºC
figure()
for T = 19:26
    plot(Simulation.Galvanostatic(T).capa, Simulation.Galvanostatic(T).T.d1(:,1))
    hold on
end
% Experimental
for i = 3:4
    for j = 1:6
        plot(flip(test_info(i).test.For_sim.Crate(j).capacity),test_info(i).test.For_sim.Crate(j).temperature+273.15,'marker','x')
        hold on
    end
end
title('Galvanostatic discharges at 298.15 K and temperature dependence')
legend('0.1C num','0.2C num','0.5C num','1C num','2C num','5C num','10C num','15C num','0.1C exp','0.2C exp','0.5C exp','1C exp','2C exp','5C exp','10C exp')
xlabel('Capacity (Ah)')
ylabel('Temperature (K)')
grid on


%% T3 = 318 k (45 ºC) discharge
figure()
% Simulation
for T = 32:39
    plot(Simulation.Galvanostatic(T).capa, Simulation.Galvanostatic(T).Ecell)
    hold on
end

% Experimental
for i = 1:2
    for j = 1:7
        plot(flip(test_info(i).test.For_sim.Crate(j).capacity),test_info(i).test.For_sim.Crate(j).voltage)
        hold on
    end
end
title('Galvanostatic discharges at 318.15 K')
legend('0.1C num','0.2C num','0.5C num','1C num','2C num','5C num','10C num','15C num','0.1C exp','0.2C exp','0.5C exp','1C exp','2C exp','5C exp','10C exp')
xlabel('Capacity (Ah)')
ylabel('Voltage (V)')
grid on

% Temperature behaviour
figure()
for T =  32:39
    plot(Simulation.Galvanostatic(T).capa, Simulation.Galvanostatic(T).T.d1(:,1))
    hold on
end
% Experimental
for i = 1:2
    for j = 1:6
        plot(flip(test_info(i).test.For_sim.Crate(j).capacity),test_info(i).test.For_sim.Crate(j).temperature+273.15,'marker','x')
        hold on
    end
end
title('Galvanostatic discharges at 308.15 K and temperature dependence')
legend('0.1C num','0.2C num','0.5C num','1C num','2C num','5C num','10C num','15C num','0.1C exp','0.2C exp','0.5C exp','1C exp','2C exp','5C exp','10C exp')
xlabel('Capacity (Ah)')
ylabel('Temperature (K)')
grid on

%% Voltage vs capacity plot (charge)
% T1 = 278.15 K (5 ºC)
% Simulation
figure()
for T = 1:5
    plot(Simulation.Galvanostatic(T).capa, Simulation.Galvanostatic(T).Ecell)
    hold on
end

% Experimental

title('Galvanostatic charges at 278.15 K')
legend('0.1C','0.2C ','0.5','1C','2C')
xlabel('Capacity (Ah)')
ylabel('Voltage (V)')
grid on

%% T2 = 298.15 K (25 ºC)
figure()
% Simulation
for T = 14:18
    plot(Simulation.Galvanostatic(T).capa, Simulation.Galvanostatic(T).Ecell)
    hold on
end

% Experimental
title('Galvanostatic charges at 298.15 K')
legend('0.1C','0.2C ','0.5','1C','2C')
xlabel('Capacity (Ah)')
ylabel('Voltage (V)')
grid on

%% T3 = 318 k (45 ºC)
figure()
% Simulation
for T = 27:31
    plot(Simulation.Galvanostatic(T).capa, Simulation.Galvanostatic(T).Ecell)
    hold on
end

% Experimental

title('Galvanostatic charges at 318.15 K')
legend('0.1C','0.2C ','0.5','1C','2C')
xlabel('Capacity (Ah)')
ylabel('Voltage (V)')
grid on

%% Error in the voltage response: Mean and max percentage and mV for all Crates,charge-discharge and temperatures

% 5 deg charge
for i = 1:5
    
Mean_per_5_cha= ((test_info(1).test.For_sim.Crate(i+8).voltage-Simulation.Galvanostatic(i).Ecell')./test_info(1).test.For_sim.Crate(i+8).voltage)*100;
% Max_per_5_cha
% Mean_mV_5_cha
% Max_mV_5_cha
end

% 5 deg discharge
% Mean_per_5_dch
% Max_per_5_dch
% Mean_mV_5_dch
% Max_mV_5_dch

% 25 deg charge
% Mean_per_25_cha
% Max_per_25_cha
% Mean_mV_25_cha
% Max_mV_25_cha

% 25 deg discharge
% Mean_per_25_dch
% Max_per_25_dch
% Mean_mV_25_dch
% Max_mV_25_dch

% 45 deg charge
% Mean_per_45_cha
% Max_per_45_cha
% Mean_mV_45_cha
% Max_mV_45_cha

% 45 deg discharge
% Mean_per_45_dch
% Max_per_45_dch
% Mean_mV_45_dch
% Max_mV_45_dch


