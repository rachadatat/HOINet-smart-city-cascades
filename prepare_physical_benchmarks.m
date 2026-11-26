%% Prepare hydraulic and power profiles for the chapter experiment
% Requires MATPOWER 8 and EPANET-Matlab Toolkit 2.2.6.1. The script uses
% the unmodified case33bw.m and Net3.inp files stored in benchmark_data.

clear; clc;
warning('off','all');

dataDir = fullfile(pwd,'benchmark_data');
epanetCandidates = {getenv('EPANET_MATLAB_TOOLKIT'), ...
    getenv('EPANET_MATLAB_TOOLKIT')};
matpowerCandidates = {getenv('MATPOWER_ROOT'), ...
    getenv('MATPOWER_ROOT')};
epanetRoot = first_existing_folder(epanetCandidates);
matpowerRoot = first_existing_folder(matpowerCandidates);
assert(~isempty(epanetRoot),'Set EPANET_MATLAB_TOOLKIT to the epanet_matlab_toolkit folder.');
assert(~isempty(matpowerRoot),'Set MATPOWER_ROOT to a MATPOWER installation folder.');

addpath(epanetRoot);
addpath(genpath(matpowerRoot));
addpath(dataDir);

%% EPANET Net3: 24 regular hourly pressure and pump-load profiles
d = epanet('Net3.inp');
h = d.getComputedHydraulicTimeSeries;
nodeNames = string(d.getNodeNameID)';
linkNames = string(d.getLinkNameID)';
linkNodes = d.getLinkNodesIndex;
pumpIndex = d.getLinkPumpIndex;
pumpNames = string(d.getLinkPumpNameID)';
pumpNodes = linkNodes(pumpIndex,:);

regular = abs(h.Time/3600-round(h.Time/3600)) < 1e-9 & h.Time < 24*3600;
hour = round(h.Time(regular)/3600);
pressure = h.Pressure(regular,:);
pumpFlowGpm = h.Flow(regular,pumpIndex);
pumpPowerKw = h.Energy(regular,pumpIndex);
pumpStatus = h.Status(regular,pumpIndex);

waterVarNames = [{'hour'},cellstr(compose('W%03d',1:numel(nodeNames)))];
waterProfileTable = array2table([hour,pressure],'VariableNames',waterVarNames);
writetable(waterProfileTable,fullfile(dataDir,'net3_hourly_pressure.csv'));

waterNodeTable = table((1:numel(nodeNames))',nodeNames, ...
    'VariableNames',{'water_index','node_id'});
writetable(waterNodeTable,fullfile(dataDir,'net3_nodes.csv'));

waterEdgeTable = table((1:numel(linkNames))',linkNames,linkNodes(:,1),linkNodes(:,2), ...
    'VariableNames',{'link_index','link_id','source_water_index','target_water_index'});
writetable(waterEdgeTable,fullfile(dataDir,'net3_edges.csv'));

d.unload;

%% IEEE 33-bus: real EV and hydraulic pump demand are added as loads
define_constants;
mpcBase = loadcase('case33bw');
basePd = mpcBase.bus(:,PD);
baseQd = mpcBase.bus(:,QD);
nBus = size(mpcBase.bus,1);
powerEdgeTable = table(mpcBase.branch(:,F_BUS),mpcBase.branch(:,T_BUS), ...
    mpcBase.branch(:,BR_STATUS), ...
    'VariableNames',{'source_bus','target_bus','status'});
writetable(powerEdgeTable,fullfile(dataDir,'ieee33_edges.csv'));

stationMeta = readtable(fullfile(dataDir,'boulder_ev_station_metadata.csv'), ...
    'TextType','string','VariableNamingRule','preserve');
evProfileLong = readtable(fullfile(dataDir,'boulder_ev_hour_of_week_profile.csv'), ...
    'TextType','string','VariableNamingRule','preserve');

stationIds = string(stationMeta.station_id);
siteIds = unique(string(stationMeta.site_id),'sorted');
siteEnergy = zeros(numel(siteIds),1);
for s = 1:numel(siteIds)
    siteEnergy(s) = sum(stationMeta.total_energy_kwh(stationMeta.site_id == siteIds(s)));
end
[~,siteRank] = sort(siteEnergy,'descend');
loadBuses = find(basePd > 0);
[~,busRankLocal] = sort(basePd(loadBuses),'descend');
rankedBuses = loadBuses(busRankLocal);
assert(numel(rankedBuses) >= numel(siteIds),'Not enough nonzero-load buses for site mapping.');
siteBus = zeros(numel(siteIds),1);
siteBus(siteRank) = rankedBuses(1:numel(siteIds));

stationBus = zeros(numel(stationIds),1);
for k = 1:numel(stationIds)
    stationBus(k) = siteBus(siteIds == stationMeta.site_id(k));
end

% Net3 pumps are assigned to the two highest-demand IEEE load buses.
pumpBus = rankedBuses(1:numel(pumpIndex));

stationMapTable = table(stationIds,string(stationMeta.site_id),string(stationMeta.Address), ...
    stationBus,stationMeta.session_count,stationMeta.total_energy_kwh, ...
    'VariableNames',{'station_id','site_id','address','supply_bus','session_count','total_energy_kwh'});
writetable(stationMapTable,fullfile(dataDir,'boulder_ev_station_bus_mapping.csv'));

siteMapTable = table(siteIds,siteBus,siteEnergy, ...
    'VariableNames',{'site_id','supply_bus','total_energy_kwh'});
writetable(siteMapTable,fullfile(dataDir,'boulder_ev_site_bus_mapping.csv'));

pumpTableRows = cell(0,10);
for p = 1:numel(pumpIndex)
    fromName = nodeNames(pumpNodes(p,1));
    toName = nodeNames(pumpNodes(p,2));
    for t = 1:numel(hour)
        pumpTableRows = [pumpTableRows; {hour(t),pumpNames(p),pumpNodes(p,1), ...
            pumpNodes(p,2),fromName,toName,pumpBus(p),pumpFlowGpm(t,p), ...
            pumpPowerKw(t,p),pumpStatus(t,p)}]; %#ok<AGROW>
    end
end
pumpTable = cell2table(pumpTableRows,'VariableNames', ...
    {'hour','pump_id','source_water_index','target_water_index','source_node_id', ...
    'target_node_id','supply_bus','flow_gpm','electrical_load_kw','status'});
writetable(pumpTable,fullfile(dataDir,'net3_pump_hourly_load.csv'));

evMean = zeros(168,numel(stationIds));
evStd = zeros(168,numel(stationIds));
for k = 1:numel(stationIds)
    rows = evProfileLong.station_id == stationIds(k);
    [sortedHour,order] = sort(evProfileLong.hour_of_week(rows));
    assert(isequal(sortedHour,(0:167)'),'Incomplete EV hour-of-week profile.');
    values = evProfileLong.mean_kw(rows);
    deviations = evProfileLong.std_kw(rows);
    evMean(:,k) = values(order);
    evStd(:,k) = deviations(order);
end

[Bbus,~,Pbusinj,~] = makeBdc(mpcBase);
powerInjectionMw = zeros(168,nBus);
voltageMagnitude = zeros(168,nBus);
evBusLoadMw = zeros(168,nBus);
pumpBusLoadMw = zeros(168,nBus);
options = mpoption('verbose',0,'out.all',0);

for how = 0:167
    evLoad = accumarray(stationBus,evMean(how+1,:)'/1000,[nBus 1],@sum,0);
    waterHour = mod(how,24);
    pumpRows = pumpTable.hour == waterHour;
    pumpLoad = accumarray(pumpTable.supply_bus(pumpRows), ...
        pumpTable.electrical_load_kw(pumpRows)/1000,[nBus 1],@sum,0);
    evBusLoadMw(how+1,:) = evLoad;
    pumpBusLoadMw(how+1,:) = pumpLoad;

    mpc = mpcBase;
    mpc.bus(:,PD) = basePd + evLoad + pumpLoad;
    mpc.bus(:,QD) = baseQd + evLoad*tan(acos(0.98)) + pumpLoad*tan(acos(0.90));
    result = runpf(mpc,options);
    assert(result.success,'IEEE 33-bus power flow failed at hour-of-week %d.',how);
    theta = deg2rad(result.bus(:,VA));
    powerInjectionMw(how+1,:) = mpcBase.baseMVA*(Bbus*theta + Pbusinj);
    voltageMagnitude(how+1,:) = result.bus(:,VM);
end

powerNames = [{'hour_of_week'},cellstr(compose('P%02d',1:nBus))];
writetable(array2table([(0:167)',powerInjectionMw],'VariableNames',powerNames), ...
    fullfile(dataDir,'ieee33_hour_of_week_injection.csv'));
writetable(array2table([(0:167)',voltageMagnitude],'VariableNames',powerNames), ...
    fullfile(dataDir,'ieee33_hour_of_week_voltage.csv'));
writetable(array2table([(0:167)',evBusLoadMw],'VariableNames',powerNames), ...
    fullfile(dataDir,'ieee33_ev_load_by_bus.csv'));
writetable(array2table([(0:167)',pumpBusLoadMw],'VariableNames',powerNames), ...
    fullfile(dataDir,'ieee33_pump_load_by_bus.csv'));
writematrix(full(Bbus),fullfile(dataDir,'ieee33_dc_measurement_matrix.csv'));

fprintf('Prepared %d power buses, %d water nodes, %d EV ports, %d sites, and %d pumps.\n', ...
    nBus,numel(nodeNames),numel(stationIds),numel(siteIds),numel(pumpIndex));

function folder = first_existing_folder(candidates)
folder = '';
for k = 1:numel(candidates)
    candidate = char(candidates{k});
    if ~isempty(candidate) && isfolder(candidate)
        folder = candidate;
        return;
    end
end
end


