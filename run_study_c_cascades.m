%% Study C: sequential interdependent cascading failures
% Clean system data:
%   Power: Baran-Wu 33-bus radial distribution benchmark
%   Water: EPANET Net3 with actual pump-outage pressure trajectories
%   EV: 29,934 valid 2023 charging sessions at 46 City of Boulder ports
% A feeder-line outage disconnects a downstream subtree. Pumps and charging
% sites supplied from that subtree then fail, creating a cross-layer sequence.
% This script is separate from the chapter benchmark and does not edit it.
clear; close all; clc;

cfg.numRuns=20; cfg.trainScenarios=160; cfg.validationScenarios=80;
cfg.testScenarios=180; cfg.trainingDropout=0.05;
cfg.hiddenUnits=14; cfg.epochs=140; cfg.learningRate=0.008;
cfg.measurementNoise=0.16; cfg.stageHours=[1 6 12];
cfg.waterImpactThresholdPsi=1.0; cfg.eventTopK=5;
cfg.incidenceShuffleSeed=20260802;
modelNames={'Pairwise neighborhoods','Shuffled incidence','Higher-order incidence'};
metricNames={'Event F1','Event AUROC','Impacted-node localization F1',...
    'Dependency-asset localization F1','Early cascade F1',...
    'Mid-stage cascade F1','Late cascade F1',...
    'Pump-involving cascade F1','EV-only cascade F1',...
    'Multi-group cascade F1','Localization AP'};
nModels=numel(modelNames); nMetrics=numel(metricNames);
scores=zeros(cfg.numRuns,nModels,nMetrics); datasetStats=zeros(cfg.numRuns,10);
assetsDir=fullfile(pwd,'study_c_results'); dataDir=fullfile(pwd,'benchmark_data');
if ~exist(assetsDir,'dir'), mkdir(assetsDir); end
hydraulicFile=fullfile(pwd,'study_c_data','study_c_hydraulic_responses.mat');
city=load_benchmark_system(dataDir,cfg.incidenceShuffleSeed,hydraulicFile);
if strcmp(getenv('STUDY_C_FIGURE_ONLY'),'1')
    create_dependent_system_figure(city,fullfile(pwd,'study_c_chapter_assets',...
        'figure_1_dependent_system_large_text.png'));
    return;
end
fprintf('Loaded %d nodes: %d power, %d water, %d observed EV ports at %d sites.\n',...
    city.nNodes,city.nPower,city.nWater,city.nEV,city.nSites);
fprintf('Using %d eligible line-outage cascades (%d pump-involving, %d EV-only).\n',...
    numel(city.outageCases),numel(city.pumpOutageCases),numel(city.evOnlyOutageCases));

for run=1:cfg.numRuns
    train=make_cascade_dataset(city,cfg.trainScenarios,23000+run,cfg.trainingDropout,cfg);
    validation=make_cascade_dataset(city,cfg.validationScenarios,24000+run,cfg.trainingDropout,cfg);
    test=make_cascade_dataset(city,cfg.testScenarios,25000+run,cfg.trainingDropout,cfg);
    cascadeScenario=test.scenarioKinds==1;
    cascadeMask=scenario_node_mask(test,cascadeScenario);
    stageScenario=cell(1,numel(cfg.stageHours)); stageMask=cell(size(stageScenario));
    for q=1:numel(cfg.stageHours)
        stageScenario{q}=cascadeScenario & test.scenarioStages==cfg.stageHours(q);
        stageMask{q}=scenario_node_mask(test,stageScenario{q});
    end
    pumpScenario=cascadeScenario & test.scenarioPumpInvolved;
    evOnlyScenario=cascadeScenario & ~test.scenarioPumpInvolved;
    multiScenario=cascadeScenario & test.scenarioGroupCounts>1;
    datasetStats(run,:)=[mean(test.scenarioLabels),mean(test.labels),...
        mean(test.labels(cascadeMask)),mean(test.dependencyLabels(cascadeMask)),...
        sum(cascadeScenario),sum(stageScenario{1}),sum(stageScenario{2}),...
        sum(stageScenario{3}),sum(pumpScenario),sum(multiScenario)];
    for m=1:nModels
        [xTrain,xVal,xTest]=standardize_views(train.views{m},...
            validation.views{m},test.views{m});
        impactNet=train_mlp(xTrain,train.labels,cfg.hiddenUnits,cfg.epochs,...
            cfg.learningRate,27000+100*run+m);
        dependencyNet=train_mlp(xTrain,train.dependencyLabels,cfg.hiddenUnits,cfg.epochs,...
            cfg.learningRate,37000+100*run+m);
        pVal=predict_mlp(impactNet,xVal);
        pDependencyVal=predict_mlp(dependencyNet,xVal);
        valMask=scenario_node_mask(validation,validation.scenarioKinds==1);
        locThreshold=select_threshold(validation.labels(valMask),pVal(valMask));
        dependencyThreshold=select_threshold(validation.dependencyLabels(valMask),...
            pDependencyVal(valMask));
        eventThreshold=select_threshold(validation.scenarioLabels,...
            aggregate_event_scores(pVal,validation.nNodes,cfg.eventTopK));
        pTest=predict_mlp(impactNet,xTest);
        pDependencyTest=predict_mlp(dependencyNet,xTest);
        pumpMask=scenario_node_mask(test,pumpScenario);
        evOnlyMask=scenario_node_mask(test,evOnlyScenario);
        multiMask=scenario_node_mask(test,multiScenario);
        mi=binary_metrics(test.labels(cascadeMask),pTest(cascadeMask),locThreshold);
        md=binary_metrics(test.dependencyLabels(cascadeMask),...
            pDependencyTest(cascadeMask),dependencyThreshold);
        mearly=binary_metrics(test.labels(stageMask{1}),pTest(stageMask{1}),locThreshold);
        mmid=binary_metrics(test.labels(stageMask{2}),pTest(stageMask{2}),locThreshold);
        mlate=binary_metrics(test.labels(stageMask{3}),pTest(stageMask{3}),locThreshold);
        mp=binary_metrics(test.labels(pumpMask),pTest(pumpMask),locThreshold);
        mev=binary_metrics(test.labels(evOnlyMask),pTest(evOnlyMask),locThreshold);
        mmulti=binary_metrics(test.labels(multiMask),pTest(multiMask),locThreshold);
        mt=binary_metrics(test.scenarioLabels,...
            aggregate_event_scores(pTest,test.nNodes,cfg.eventTopK),eventThreshold);
        scores(run,m,:)=[mt.f1,mt.auroc,mi.f1,md.f1,mearly.f1,mmid.f1,...
            mlate.f1,mp.f1,mev.f1,mmulti.f1,mi.ap];
    end
    fprintf('  independent run %d/%d complete\n',run,cfg.numRuns);
end

scoreMean=squeeze(mean(scores,1)); scoreStd=squeeze(std(scores,0,1));
summaryTable=table(string(modelNames'),'VariableNames',{'Model'});
for k=1:nMetrics
    baseName=matlab.lang.makeValidName(metricNames{k});
    summaryTable.([baseName 'Mean'])=scoreMean(:,k);
    summaryTable.([baseName 'Std'])=scoreStd(:,k);
end
writetable(summaryTable,fullfile(assetsDir,'study_c_summary.csv'));

% Impacted-node localization is the prespecified primary comparison.
sigIndex=3:11;
significanceTable=paired_comparison_table(scores,modelNames,metricNames,...
    1,3,sigIndex,'Higher-order minus pairwise');
writetable(significanceTable,fullfile(assetsDir,'study_c_significance.csv'));
ablationTable=paired_comparison_table(scores,modelNames,metricNames,...
    2,3,sigIndex,'Higher-order minus shuffled incidence');
writetable(ablationTable,fullfile(assetsDir,'study_c_ablation_significance.csv'));

datasetTable=array2table([(1:cfg.numRuns)' datasetStats],...
    'VariableNames',{'Run','EventPrevalence','ImpactPrevalenceAllScenarios',...
    'ImpactPrevalenceCascadeScenarios','DependencyPrevalenceCascadeScenarios',...
    'CascadeScenarios','EarlyCascadeScenarios','MidCascadeScenarios',...
    'LateCascadeScenarios','PumpInvolvingCascadeScenarios','MultiGroupCascadeScenarios'});
writetable(datasetTable,fullfile(assetsDir,'study_c_dataset_stats.csv'));
runNumber=[]; runModel={}; runMetric={}; runValue=[];
for run=1:cfg.numRuns
    for m=1:nModels
        for k=1:nMetrics
            runNumber(end+1,1)=run; runModel{end+1,1}=modelNames{m};
            runMetric{end+1,1}=metricNames{k}; runValue(end+1,1)=scores(run,m,k);
        end
    end
end
writetable(table(runNumber,runModel,runMetric,runValue,...
    'VariableNames',{'Run','Model','Metric','Value'}),...
    fullfile(assetsDir,'study_c_runs.csv'));
benchmarkTable=table(city.nPower,city.nWater,city.nEV,city.nSites,city.nPumps,...
    city.nHyperedges,city.nPhysicalEdges,city.nPairwiseEdges,numel(city.outageCases),...
    numel(city.pumpOutageCases),numel(city.evOnlyOutageCases),...
    'VariableNames',{'PowerBuses','WaterNodes','EVPorts','EVSites','WaterPumps',...
    'DependencyGroups','PhysicalPairwiseEdges','PairwiseEdgesAfterCliqueExpansion',...
    'EligibleLineOutages','PumpInvolvingLineOutages','EVOnlyLineOutages'});
writetable(benchmarkTable,fullfile(assetsDir,'study_c_benchmark_stats.csv'));
save(fullfile(assetsDir,'study_c_results.mat'),'cfg','modelNames','metricNames',...
    'scores','scoreMean','scoreStd','datasetStats','benchmarkTable',...
    'significanceTable','ablationTable');
create_performance_figure(scoreMean,scoreStd,modelNames,...
    fullfile(assetsDir,'study_c_performance.png'));
disp(summaryTable); disp(significanceTable); disp(ablationTable); disp(benchmarkTable);

function city=load_benchmark_system(dataDir,shuffleSeed,hydraulicFile)
powerEdgeTable=readtable(fullfile(dataDir,'ieee33_edges.csv'));
powerEdges=[powerEdgeTable.source_bus(powerEdgeTable.status>0) powerEdgeTable.target_bus(powerEdgeTable.status>0)];
powerH=readmatrix(fullfile(dataDir,'ieee33_dc_measurement_matrix.csv')); nP=size(powerH,1);
waterNodes=readtable(fullfile(dataDir,'net3_nodes.csv'),'TextType','string');
waterLinks=readtable(fullfile(dataDir,'net3_edges.csv'),'TextType','string');
nW=height(waterNodes); waterEdges=[waterLinks.source_water_index waterLinks.target_water_index];
stationMap=readtable(fullfile(dataDir,'boulder_ev_station_bus_mapping.csv'),'TextType','string');
siteMap=readtable(fullfile(dataDir,'boulder_ev_site_bus_mapping.csv'),'TextType','string');
evLinks=readtable(fullfile(dataDir,'boulder_ev_edges.csv'),'TextType','string');
nE=height(stationMap); stationIDs=stationMap.station_id;
siteIDs=unique(stationMap.site_id,'stable'); nSites=numel(siteIDs);
pTable=readtable(fullfile(dataDir,'ieee33_hour_of_week_injection.csv'));
pProfile=table2array(pTable(:,2:end))';
wTable=readtable(fullfile(dataDir,'net3_hourly_pressure.csv'));
wProfile=repmat(table2array(wTable(:,2:end))',1,7);
evLong=readtable(fullfile(dataDir,'boulder_ev_hour_of_week_profile.csv'),'TextType','string');
eProfile=zeros(nE,168);
for e=1:nE
    rows=evLong.station_id==stationIDs(e);
    eProfile(e,evLong.hour_of_week(rows)+1)=evLong.mean_kw(rows);
end
raw=[pProfile;wProfile;eProfile]; mu=mean(raw,2); sigma=std(raw,0,2);
ranges={1:nP,nP+(1:nW),nP+nW+(1:nE)};
for q=1:3
    s=sigma(ranges{q}); s=s(s>1e-8);
    if isempty(s), floorValue=1; else, floorValue=max(0.15*median(s),1e-3); end
    sigma(ranges{q})=max(sigma(ranges{q}),floorValue);
end
cleanZ=(raw-mu)./sigma; n=nP+nW+nE; physical=zeros(n);
for k=1:size(powerEdges,1), physical(powerEdges(k,1),powerEdges(k,2))=1; end
for k=1:size(waterEdges,1)
    i=nP+waterEdges(k,1); j=nP+waterEdges(k,2); physical(i,j)=1;
end
for k=1:height(evLinks)
    i=find(stationIDs==evLinks.source(k),1); j=find(stationIDs==evLinks.target(k),1);
    if ~isempty(i)&&~isempty(j), physical(nP+nW+i,nP+nW+j)=1; end
end
physical=double((physical+physical')>0); physical(1:n+1:end)=0;
pumpTable=readtable(fullfile(dataDir,'net3_pump_hourly_load.csv'),'TextType','string');
[~,pumpRows]=unique(pumpTable.pump_id,'stable'); pumpInfo=pumpTable(pumpRows,:);
groups=cell(0,1); groupKinds=strings(0,1); groupNames=strings(0,1);
for p=1:height(pumpInfo)
    groups{end+1,1}=[pumpInfo.supply_bus(p),nP+pumpInfo.source_water_index(p),...
        nP+pumpInfo.target_water_index(p)];
    groupKinds(end+1,1)="pump"; groupNames(end+1,1)="Pump "+pumpInfo.pump_id(p);
end
for s=1:nSites
    members=find(stationMap.site_id==siteIDs(s)); row=find(siteMap.site_id==siteIDs(s),1);
    groups{end+1,1}=[siteMap.supply_bus(row),nP+nW+members'];
    groupKinds(end+1,1)="charging"; groupNames(end+1,1)=siteIDs(s);
end
H=zeros(n,numel(groups)); adjacency=physical;
for g=1:numel(groups)
    idx=unique(groups{g}); H(idx,g)=1; adjacency(idx,idx)=1;
end
adjacency(1:n+1:end)=0; adjacency=double(adjacency>0);

% Eligible initiating outages are closed feeder lines whose downstream island
% contains one to six dependency groups and at most 12 power buses. This omits
% near-systemwide disconnections while retaining pump and EV cascades.
powerGraph=graph(powerEdges(:,1),powerEdges(:,2),[],nP);
groupSupplyBuses=cellfun(@(x)x(1),groups);
outageCases=struct('edge',{},'disconnectedBuses',{},'groupIndices',{},'pumpInvolved',{});
for e=1:size(powerEdges,1)
    trial=rmedge(powerGraph,powerEdges(e,1),powerEdges(e,2));
    component=conncomp(trial); disconnected=find(component~=component(1));
    groupIndices=find(ismember(groupSupplyBuses,disconnected));
    if isempty(groupIndices)||numel(groupIndices)>6||numel(disconnected)>12, continue; end
    item.edge=powerEdges(e,:); item.disconnectedBuses=disconnected(:)';
    item.groupIndices=groupIndices(:)';
    item.pumpInvolved=any(ismember(pumpInfo.supply_bus,disconnected));
    outageCases(end+1)=item; %#ok<AGROW>
end
pumpOutageCases=find([outageCases.pumpInvolved]);
evOnlyOutageCases=find(~[outageCases.pumpInvolved]);
assert(~isempty(pumpOutageCases)&&~isempty(evOnlyOutageCases),...
    'Study C needs both pump-involving and EV-only outage cases.');
hydraulic=load(hydraulicFile);
assert(size(hydraulic.pressure,1)==4&&size(hydraulic.pressure,2)==24&&...
    size(hydraulic.pressure,3)==nW,'Unexpected Study C hydraulic-response dimensions.');

pairOp=row_normalize(adjacency+eye(n));
triW=adjacency.*(adjacency*adjacency); triDegree=sum(triW,2)/2;
if max(triDegree)>0, triOp=row_normalize(triW+eye(n)); else, triOp=pairOp; end
groupSize=sum(H,1)'; nodeGroupDegree=sum(H,2);
higherOp=diag(1./max(nodeGroupDegree,1))*H*diag(1./groupSize)*H';

% Negative control: preserve the number and size of dependency groups but
% randomize memberships within each infrastructure. This has the same feature
% dimension as the true incidence view but destroys the correct dependencies.
rngState=rng; rng(shuffleSeed,'twister'); Hshuffled=H;
Hshuffled(1:nP,:)=H(randperm(nP),:);
waterRows=nP+(1:nW); Hshuffled(waterRows,:)=H(nP+randperm(nW),:);
evRows=nP+nW+(1:nE); Hshuffled(evRows,:)=H(nP+nW+randperm(nE),:);
rng(rngState);
shuffledGroupSize=sum(Hshuffled,1)';
shuffledNodeDegree=sum(Hshuffled,2);
shuffledHigherOp=diag(1./max(shuffledNodeDegree,1))*Hshuffled*...
    diag(1./max(shuffledGroupSize,1))*Hshuffled';
xy=zeros(n,2); xy(1:nP,:)=band_coordinates(physical(1:nP,1:nP),2.55,3.25);
wi=nP+(1:nW); xy(wi,:)=band_coordinates(physical(wi,wi),1.25,2.05);
for s=1:nSites
    members=find(stationMap.site_id==siteIDs(s)); x=0.04+0.92*(s-1)/max(nSites-1,1);
    xy(nP+nW+members,1)=x+0.006*((1:numel(members))-mean(1:numel(members)))';
    xy(nP+nW+members,2)=linspace(0.22,0.74,numel(members));
end
types=[ones(nP,1);2*ones(nW,1);3*ones(nE,1)];
D=[xy(:,1),xy(:,2)/3.25,types==1,types==2,types==3,sum(adjacency,2)/max(sum(adjacency,2))];
d2=zeros(n); for d=1:size(D,2), d2=d2+(D(:,d)-D(:,d)').^2; end
graphonOp=row_normalize(exp(-d2/(2*0.34^2))+0.05*eye(n));
city.nNodes=n; city.nPower=nP; city.nWater=nW; city.nEV=nE; city.nSites=nSites;
city.nPumps=height(pumpInfo); city.nHyperedges=numel(groups);
city.nPhysicalEdges=nnz(triu(physical)); city.nPairwiseEdges=nnz(triu(adjacency));
city.cleanZ=cleanZ; city.cleanRaw=raw; city.cleanMu=mu; city.cleanSigma=sigma; city.nodeType=types; city.physicalAdjacency=physical;
city.adjacency=adjacency; city.pairwiseOperator=pairOp;
city.triangleOperator=triOp; city.triangleDegree=triDegree;
city.incidence=H; city.higherOrderOperator=higherOp;
city.shuffledIncidence=Hshuffled; city.shuffledHigherOrderOperator=shuffledHigherOp;
city.graphonOperator=graphonOp;
city.groups=groups; city.groupKinds=groupKinds; city.groupNames=groupNames;
city.powerH=powerH;
city.coordinates=xy; city.stationIDs=stationIDs; city.stationMap=stationMap;
city.siteIDs=siteIDs; city.pumpInfo=pumpInfo;
city.hydraulicPressure=hydraulic.pressure;
city.outageCases=outageCases; city.pumpOutageCases=pumpOutageCases;
city.evOnlyOutageCases=evOnlyOutageCases; city.pumpSupplyBuses=pumpInfo.supply_bus;
end

function xy=band_coordinates(A,yMin,yMax)
n=size(A,1); L=diag(sum(A,2))-A; [V,E]=eig(L,'vector'); [~,o]=sort(E);
if n>=3, x=V(:,o(2)); y=V(:,o(3)); else, x=(1:n)'; y=zeros(n,1); end
xy=[rescale_safe(x,0.04,0.96),rescale_safe(y,yMin,yMax)];
end
function y=rescale_safe(x,a,b)
if max(x)-min(x)<eps, y=linspace(a,b,numel(x))'; else, y=a+(b-a)*(x-min(x))/(max(x)-min(x)); end
end


function out=make_cascade_dataset(city,nScenarios,seed,dropRate,cfg)
rng(seed,'twister'); n=city.nNodes; views=cell(1,3);
for m=1:3, views{m}=zeros(n*nScenarios,16); end
labels=zeros(n*nScenarios,1); dependencyLabels=zeros(n*nScenarios,1);
scenarioLabels=zeros(nScenarios,1); scenarioKinds=zeros(nScenarios,1);
scenarioStages=zeros(nScenarios,1); scenarioPumpInvolved=false(nScenarios,1);
scenarioGroupCounts=zeros(nScenarios,1);
typeOneHot=[city.nodeType==1 city.nodeType==2 city.nodeType==3];
degreeNorm=sum(city.adjacency,2)/max(max(sum(city.adjacency,2)),1);
waterRows=city.nPower+(1:city.nWater);
for t=1:nScenarios
    stage=cfg.stageHours(randi(numel(cfg.stageHours)));
    day=randi(7); hour=(day-1)*24+stage+1; prev=mod(hour-2,168)+1;
    failedRaw=city.cleanRaw(:,hour); y=zeros(n,1); dependencyY=zeros(n,1);
    if rand<0.35
        scenarioKind=0;
    else
        scenarioKind=1;
        % Equal sampling of pump-involving and EV-only initiating lines keeps
        % both cross-sector paths represented despite only two water pumps.
        if rand<0.5
            pool=city.pumpOutageCases;
        else
            pool=city.evOnlyOutageCases;
        end
        outageIndex=pool(randi(numel(pool))); item=city.outageCases(outageIndex);
        disconnected=item.disconnectedBuses;
        failedRaw(disconnected)=0; y(disconnected)=1;
        for g=item.groupIndices
            nodes=city.groups{g}; dependencyY(nodes)=1;
            if city.groupKinds(g)=="charging"
                failedRaw(nodes(2:end))=0;
            end
        end
        failedPump=ismember(city.pumpSupplyBuses,disconnected);
        hydraulicPattern=1+double(failedPump(1))+2*double(failedPump(2));
        if hydraulicPattern>1
            normalPressure=squeeze(city.hydraulicPressure(1,stage+1,:));
            failedPressure=squeeze(city.hydraulicPressure(hydraulicPattern,stage+1,:));
            failedRaw(waterRows)=failedPressure;
            impactedWater=abs(failedPressure-normalPressure)>=cfg.waterImpactThresholdPsi;
            y(waterRows(impactedWater))=1;
        end
        y=max(y,dependencyY);
        scenarioPumpInvolved(t)=item.pumpInvolved;
        scenarioGroupCounts(t)=numel(item.groupIndices);
    end
    scenarioKinds(t)=scenarioKind; scenarioLabels(t)=scenarioKind>0;
    scenarioStages(t)=stage;
    measurement=(failedRaw-city.cleanMu)./city.cleanSigma+...
        cfg.measurementNoise*randn(n,1);
    rate=measurement-city.cleanZ(:,prev)+0.10*randn(n,1);
    missing=rand(n,1)<dropRate; measurement(missing)=0; rate(missing)=0;
    forecast=city.cleanZ(:,hour); residual=measurement-forecast;
    nm=city.pairwiseOperator*measurement; nr=city.pairwiseOperator*rate;
    pMean=city.pairwiseOperator*residual;
    pAbs=city.pairwiseOperator*abs(residual);
    pSecond=city.pairwiseOperator*(residual.^2);
    pStd=sqrt(max(pSecond-pMean.^2,0));
    pCoherence=abs(pMean)./max(pAbs,1e-8);
    [sgMean,sgAbs,sgStd,sgCoherence]=incidence_node_features(...
        city.shuffledIncidence,residual);
    [gMean,gAbs,gStd,gCoherence]=incidence_node_features(city.incidence,residual);
    local=[measurement rate forecast residual double(missing) typeOneHot];
    base=[local nm nr abs(measurement-nm)];
    pair=[base pMean pAbs pStd pCoherence degreeNorm];
    shuffled=[base sgMean sgAbs sgStd sgCoherence ...
        double(sum(city.shuffledIncidence,2)>0)];
    higher=[base gMean gAbs gStd gCoherence ...
        double(sum(city.incidence,2)>0)];
    rows=(t-1)*n+(1:n); views{1}(rows,:)=pair;
    views{2}(rows,:)=shuffled; views{3}(rows,:)=higher;
    labels(rows)=y; dependencyLabels(rows)=dependencyY;
end
out.views=views; out.labels=labels; out.dependencyLabels=dependencyLabels;
out.scenarioLabels=scenarioLabels; out.scenarioKinds=scenarioKinds;
out.scenarioStages=scenarioStages; out.scenarioPumpInvolved=scenarioPumpInvolved;
out.scenarioGroupCounts=scenarioGroupCounts; out.nNodes=n;
end

function [nodeMean,nodeAbs,nodeStd,nodeCoherence]=incidence_node_features(H,residual)
groupSize=max(sum(H,1)',1);
groupMean=(H'*residual)./groupSize;
groupAbs=(H'*abs(residual))./groupSize;
groupSecond=(H'*(residual.^2))./groupSize;
groupStd=sqrt(max(groupSecond-groupMean.^2,0));
groupCoherence=abs(groupMean)./max(groupAbs,1e-8);
n=size(H,1); active=sum(H,2)>0;
candidate=repmat(groupAbs',n,1); candidate(H==0)=-Inf;
[~,selectedGroup]=max(candidate,[],2); selectedGroup(~active)=1;
nodeMean=groupMean(selectedGroup); nodeAbs=groupAbs(selectedGroup);
nodeStd=groupStd(selectedGroup); nodeCoherence=groupCoherence(selectedGroup);
nodeMean(~active)=0; nodeAbs(~active)=0;
nodeStd(~active)=0; nodeCoherence(~active)=0;
end

function [trainOut,varargout]=standardize_views(trainIn,varargin)
mu=mean(trainIn,1); sigma=std(trainIn,0,1); sigma(sigma<1e-8)=1;
trainOut=(trainIn-mu)./sigma; varargout=cell(1,numel(varargin));
for k=1:numel(varargin), varargout{k}=(varargin{k}-mu)./sigma; end
end
function net=train_mlp(x,y,nHidden,nEpochs,rate,seed)
rng(seed,'twister'); pos=find(y==1); neg=find(y==0);
neg=neg(randperm(numel(neg),min(numel(neg),2*numel(pos))));
keep=[pos;neg]; keep=keep(randperm(numel(keep))); x=x(keep,:); y=y(keep);
[nSamples,nFeatures]=size(x);
W1=randn(nFeatures,nHidden)*sqrt(2/max(nFeatures,1)); b1=zeros(1,nHidden);
W2=randn(nHidden,1)*sqrt(2/max(nHidden,1)); b2=0;
mW1=zeros(size(W1)); vW1=mW1; mb1=zeros(size(b1)); vb1=mb1;
mW2=zeros(size(W2)); vW2=mW2; mb2=0; vb2=0; beta1=.9; beta2=.999; epsilon=1e-8;
for epoch=1:nEpochs
    z1=x*W1+b1; h=max(z1,0); p=sigmoid(h*W2+b2); d=(p-y)/nSamples;
    dW2=h'*d+1e-4*W2; db2=sum(d); dz=(d*W2').*double(z1>0);
    dW1=x'*dz+1e-4*W1; db1=sum(dz,1);
    [W1,mW1,vW1]=adam_step(W1,dW1,mW1,vW1,epoch,rate,beta1,beta2,epsilon);
    [b1,mb1,vb1]=adam_step(b1,db1,mb1,vb1,epoch,rate,beta1,beta2,epsilon);
    [W2,mW2,vW2]=adam_step(W2,dW2,mW2,vW2,epoch,rate,beta1,beta2,epsilon);
    [b2,mb2,vb2]=adam_step(b2,db2,mb2,vb2,epoch,rate,beta1,beta2,epsilon);
end
net.W1=W1; net.b1=b1; net.W2=W2; net.b2=b2;
end
function [p,m,v]=adam_step(p,g,m,v,t,rate,b1,b2,e)
m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*(g.^2);
p=p-rate*(m/(1-b1^t))./(sqrt(v/(1-b2^t))+e);
end
function p=predict_mlp(net,x), p=sigmoid(max(x*net.W1+net.b1,0)*net.W2+net.b2); end
function out=sigmoid(x)
out=zeros(size(x)); q=x>=0; out(q)=1./(1+exp(-x(q))); ex=exp(x(~q)); out(~q)=ex./(1+ex);
end
function threshold=select_threshold(y,p)
threshold=.5; best=-inf;
for value=.03:.01:.97
    m=binary_metrics(y,p,value);
    if m.f1>best, best=m.f1; threshold=value; end
end
end
function m=binary_metrics(y,p,threshold)
pred=p>=threshold; tp=sum(pred==1&y==1); fp=sum(pred==1&y==0); fn=sum(pred==0&y==1);
precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1);
m.f1=2*precision*recall/max(precision+recall,eps);
m.auroc=calculate_auroc(y,p); m.ap=calculate_average_precision(y,p);
end
function mask=scenario_node_mask(dataset,scenarioMask)
mask=repelem(logical(scenarioMask(:)),dataset.nNodes);
end
function s=aggregate_event_scores(p,nNodes,topK)
x=reshape(p,nNodes,[]); x=sort(x,1,'descend'); s=mean(x(1:min(topK,nNodes),:),1)';
end
function area=calculate_average_precision(y,p)
[~,o]=sort(p,'descend'); z=y(o)==1; positives=sum(z);
if positives==0, area=NaN; return; end
precision=cumsum(z)./(1:numel(z))'; area=sum(precision(z))/positives;
end
function area=calculate_auroc(y,p)
[~,o]=sort(p,'descend'); z=y(o); positives=sum(z==1); negatives=sum(z==0);
if positives==0||negatives==0, area=NaN; return; end
tpr=[0;cumsum(z==1)/positives;1]; fpr=[0;cumsum(z==0)/negatives;1]; area=trapz(fpr,tpr);
end
function y=row_normalize(x)
s=sum(x,2); s(s==0)=1; y=x./s;
end


function result=paired_comparison_table(scores,modelNames,metricNames,...
    baselineIndex,targetIndex,metricIndices,comparisonLabel)
n=numel(metricIndices); Metric=string(metricNames(metricIndices))';
Comparison=repmat(string(comparisonLabel),n,1);
BaselineModel=repmat(string(modelNames{baselineIndex}),n,1);
TargetModel=repmat(string(modelNames{targetIndex}),n,1);
BaselineMean=zeros(n,1); TargetMean=zeros(n,1); MeanDifference=zeros(n,1);
CILower=zeros(n,1); CIUpper=zeros(n,1); PairedTPValue=zeros(n,1);
SignedRankPValue=zeros(n,1); StandardizedEffectDz=zeros(n,1);
TargetWins=zeros(n,1);
for q=1:n
    baseline=scores(:,baselineIndex,metricIndices(q));
    target=scores(:,targetIndex,metricIndices(q));
    difference=target-baseline;
    [~,PairedTPValue(q),ci]=ttest(difference);
    try
        SignedRankPValue(q)=signrank(target,baseline);
    catch
        SignedRankPValue(q)=NaN;
    end
    BaselineMean(q)=mean(baseline); TargetMean(q)=mean(target);
    MeanDifference(q)=mean(difference); CILower(q)=ci(1); CIUpper(q)=ci(2);
    StandardizedEffectDz(q)=MeanDifference(q)/max(std(difference),eps);
    TargetWins(q)=sum(difference>0);
end
HolmAdjustedPValue=holm_adjust(PairedTPValue);
PrespecifiedPrimary=Metric=="Impacted-node localization F1";
result=table(Comparison,Metric,BaselineModel,TargetModel,BaselineMean,...
    TargetMean,MeanDifference,CILower,CIUpper,PairedTPValue,...
    HolmAdjustedPValue,SignedRankPValue,StandardizedEffectDz,...
    TargetWins,PrespecifiedPrimary);
end

function adjusted=holm_adjust(p)
m=numel(p); [sortedP,order]=sort(p); adjustedSorted=zeros(size(sortedP));
running=0;
for k=1:m
    running=max(running,(m-k+1)*sortedP(k));
    adjustedSorted(k)=min(running,1);
end
adjusted=zeros(size(p)); adjusted(order)=adjustedSorted;
end

function create_representation_figure(filename)
fig=figure('Color','w','Position',[100 100 1500 480]); axes('Position',[0 0 1 1],'Visible','off');
titles={'Pairwise graph','Graphlet','Graphon','Hypergraph','Simplicial complex'};
subtitles={'Direct two-node links','Repeated small pattern','Smooth large-network model',...
    'One relation joining a group','Connections with enclosed regions'};
colors=[.16 .42 .67;.29 .60 .45;.50 .36 .70;.88 .49 .18;.73 .22 .27];
for k=1:5
    x0=.02+(k-1)*.195;
    annotation('rectangle',[x0 .10 .175 .80],'Color',colors(k,:),'LineWidth',2);
    annotation('textbox',[x0+.01 .77 .155 .10],'String',titles{k},...
        'HorizontalAlignment','center','LineStyle','none','FontName','Times New Roman',...
        'FontSize',15,'FontWeight','bold','Color',colors(k,:));
    annotation('textbox',[x0+.01 .13 .155 .11],'String',subtitles{k},...
        'HorizontalAlignment','center','LineStyle','none','FontName','Times New Roman','FontSize',10.5);
    ax=axes('Position',[x0+.025 .31 .125 .38]); hold(ax,'on'); axis(ax,[0 1 0 1]); axis(ax,'off');
    nodes=[.18 .25;.50 .78;.82 .28;.50 .43];
    if k==1
        draw_edges(ax,nodes,[1 2;2 3;1 4;4 3],[.45 .45 .45],1.8);
    elseif k==2
        draw_edges(ax,nodes,[1 2;2 3;3 1;1 4;4 3],colors(k,:),2);
    elseif k==3
        [xx,yy]=meshgrid(linspace(.08,.92,18)); zz=.15+.70*exp(-((xx-yy).^2)/.09);
        imagesc(ax,[.08 .92],[.08 .92],zz); set(ax,'YDir','normal'); colormap(ax,parula);
    elseif k==4
        patch(ax,[.12 .50 .88 .50],[.28 .88 .30 .12],colors(k,:),...
            'FaceAlpha',.22,'EdgeColor',colors(k,:),'LineWidth',2);
    else
        patch(ax,nodes(1:3,1),nodes(1:3,2),colors(k,:),'FaceAlpha',.24,...
            'EdgeColor',colors(k,:),'LineWidth',2);
        draw_edges(ax,nodes,[1 4;3 4],colors(k,:),1.6);
    end
    if k~=3, scatter(ax,nodes(:,1),nodes(:,2),90,'w','filled',...
            'MarkerEdgeColor',colors(k,:),'LineWidth',2); end
end
exportgraphics(fig,filename,'Resolution',600); close(fig);
end

function create_pipeline_figure(filename)
fig=figure('Color','w','Position',[100 100 1500 480]); axes('Position',[0 0 1 1],'Visible','off');
labels={'Measured city data','Build relations','Train the AI model','AI task','Operational response'};
details={'power | water | EV charging','links | patterns | groups','fit | validate | test',...
    'detect attack | locate assets','verify | isolate | recover'};
colors=[.10 .45 .70;.22 .62 .56;.49 .39 .72;.91 .55 .15;.74 .25 .28];
for k=1:5
    x=.025+(k-1)*.195;
    annotation('textbox',[x .31 .16 .39],'String',{labels{k},'',details{k}},...
        'HorizontalAlignment','center','VerticalAlignment','middle','FontName','Times New Roman',...
        'FontSize',12,'FontWeight','bold','Color',[.12 .12 .12],...
        'EdgeColor',colors(k,:),'LineWidth',2.5,'BackgroundColor',[.98 .98 .98]);
    if k<5, annotation('arrow',[x+.162 x+.190],[.505 .505],'Color',[.3 .3 .3],'LineWidth',2); end
end
annotation('arrow',[.89 .11],[.22 .22],'Color',[.35 .35 .35],'LineWidth',1.6,'HeadStyle','vback2');
annotation('textbox',[.35 .235 .30 .055],'String','monitoring, review, and retraining',...
    'HorizontalAlignment','center','LineStyle','none','FontName','Times New Roman','FontSize',11);
exportgraphics(fig,filename,'Resolution',600); close(fig);
end

function create_dependent_system_figure(city,filename)
fig=figure('Color','w','Position',[100 100 1900 1120]); ax=axes(fig); hold(ax,'on');
rectangle(ax,'Position',[.01 2.42 .98 .93],'FaceColor',[.91 .95 .99],...
    'EdgeColor',[.25 .48 .70],'LineWidth',1.2);
rectangle(ax,'Position',[.01 1.12 .98 1.03],'FaceColor',[.91 .98 .95],...
    'EdgeColor',[.25 .60 .45],'LineWidth',1.2);
rectangle(ax,'Position',[.01 .08 .98 .84],'FaceColor',[1 .96 .89],...
    'EdgeColor',[.88 .52 .16],'LineWidth',1.2);
text(ax,.02,3.27,'33-bus power distribution benchmark','FontName','Times New Roman',...
    'FontSize',17,'FontWeight','bold');
text(ax,.02,2.07,'EPANET Net3 water network (97 nodes; 2 pumps)',...
    'FontName','Times New Roman','FontSize',17,'FontWeight','bold');
text(ax,.02,.84,'Boulder EV charging network (46 observed ports at 22 sites)',...
    'FontName','Times New Roman','FontSize',17,'FontWeight','bold');
xy=city.coordinates; p=1:city.nPower; w=city.nPower+(1:city.nWater);
e=city.nPower+city.nWater+(1:city.nEV);
draw_adjacency(ax,xy(p,:),city.physicalAdjacency(p,p),[.62 .69 .76],.7);
draw_adjacency(ax,xy(w,:),city.physicalAdjacency(w,w),[.65 .73 .68],.55);
draw_adjacency(ax,xy(e,:),city.physicalAdjacency(e,e),[.76 .70 .58],.5);
for g=1:city.nHyperedges
    nodes=city.groups{g}; source=xy(nodes(1),:); target=mean(xy(nodes(2:end),:),1);
    if city.groupKinds(g)=="pump", color=[.72 .20 .24]; width=2.2;
    else, color=[.76 .40 .08]; width=.9; end
    quiver(ax,source(1),source(2),.94*(target(1)-source(1)),.94*(target(2)-source(2)),0,...
        'Color',color,'LineWidth',width,'MaxHeadSize',.045,'AutoScale','off');
end
scatter(ax,xy(p,1),xy(p,2),54,[.13 .43 .70],'filled','MarkerEdgeColor','w','LineWidth',.6);
scatter(ax,xy(w,1),xy(w,2),28,[.20 .63 .48],'filled','MarkerEdgeColor','w','LineWidth',.4);
scatter(ax,xy(e,1),xy(e,2),42,[.91 .55 .13],'filled','MarkerEdgeColor','w','LineWidth',.5);
for q=1:city.nPumps
    nodes=city.groups{q}(2:end);
    scatter(ax,xy(nodes,1),xy(nodes,2),70,[.75 .18 .22],'s','filled','MarkerEdgeColor','w');
end
for b=1:city.nPower
    text(ax,xy(b,1),xy(b,2)+.025,sprintf('P%d',b),'FontName','Times New Roman',...
        'FontSize',9,'FontWeight','bold','HorizontalAlignment','center','Color',[.05 .19 .32]);
end
for s=1:city.nSites
    members=find(city.stationMap.site_id==city.siteIDs(s));
    pos=mean(xy(city.nPower+city.nWater+members,:),1);
    text(ax,pos(1),.12,char(city.siteIDs(s)),'Rotation',55,'FontName','Times New Roman',...
        'FontSize',9,'HorizontalAlignment','right','Color',[.43 .25 .04]);
end
h1=plot(ax,nan,nan,'-','Color',[.62 .66 .67],'LineWidth',1.2);
h2=plot(ax,nan,nan,'-','Color',[.72 .20 .24],'LineWidth',2.2);
h3=plot(ax,nan,nan,'-','Color',[.76 .40 .08],'LineWidth',1.2);
h4=scatter(ax,nan,nan,42,[.13 .43 .70],'filled');
h5=scatter(ax,nan,nan,28,[.20 .63 .48],'filled');
h6=scatter(ax,nan,nan,33,[.91 .55 .13],'filled');
legend(ax,[h1 h2 h3 h4 h5 h6],{'physical pairwise link',...
    'power supply to pump group','power supply to charging-site group',...
    'power bus','water node','observed charging port'},...
    'Location','southoutside','Orientation','horizontal','NumColumns',3,...
    'FontName','Times New Roman','FontSize',14);
axis(ax,[0 1 0 3.38]); axis(ax,'off');
exportgraphics(fig,filename,'Resolution',600); close(fig);
end

function draw_adjacency(ax,xy,A,color,width)
[i,j]=find(triu(A,1));
for k=1:numel(i), plot(ax,xy([i(k) j(k)],1),xy([i(k) j(k)],2),...
        '-','Color',color,'LineWidth',width); end
end

function create_performance_figure(scoreMean,scoreStd,modelNames,filename)
fig=figure('Color','w','Position',[100 100 1350 690]); ax=axes(fig); hold(ax,'on');
values=100*scoreMean(:,1:4)'; errors=100*scoreStd(:,1:4)';
b=bar(ax,values,'grouped','LineWidth',.8); colors=lines(numel(modelNames));
for m=1:numel(b)
    b(m).FaceColor=colors(m,:);
    errorbar(ax,b(m).XEndPoints,values(:,m),errors(:,m),'k',...
        'LineStyle','none','LineWidth',.9,'CapSize',5);
end
set(ax,'XTick',1:4,'XTickLabel',{'Event F1','Event AUROC',...
    'Impacted-node localization F1','Dependency-asset localization F1'},...
    'FontName','Times New Roman','FontSize',12);
ylabel(ax,'score (%)','FontName','Times New Roman','FontSize',12);
ylim(ax,[0 100]); grid(ax,'on'); ax.XGrid='off';
legend(ax,modelNames,'Location','southoutside','Orientation','horizontal',...
    'FontName','Times New Roman');
exportgraphics(fig,filename,'Resolution',600); close(fig);
end

function create_robustness_figure(scoreMean,scoreStd,modelNames,filename)
fig=figure('Color','w','Position',[100 100 1450 690]); ax=axes(fig); hold(ax,'on');
indices=[5 6 7]; selectedModels=[2 5 6];
values=100*scoreMean(selectedModels,indices)';
errors=100*scoreStd(selectedModels,indices)';
b=bar(ax,values,'grouped','LineWidth',.8); colors=lines(numel(selectedModels));
for m=1:numel(b)
    b(m).FaceColor=colors(m,:);
    errorbar(ax,b(m).XEndPoints,values(:,m),errors(:,m),'k',...
        'LineStyle','none','LineWidth',.9,'CapSize',5);
end
set(ax,'XTick',1:3,'XTickLabel',{'Low severity','Medium severity','High severity'},...
    'FontName','Times New Roman','FontSize',11);
ylabel(ax,'score (%)','FontName','Times New Roman','FontSize',12);
ylim(ax,[0 100]); grid(ax,'on'); ax.XGrid='off';
legend(ax,modelNames(selectedModels),'Location','southoutside',...
    'Orientation','horizontal','FontName','Times New Roman');
exportgraphics(fig,filename,'Resolution',600); close(fig);
end

function draw_edges(ax,nodes,edges,color,width)
for k=1:size(edges,1), plot(ax,nodes(edges(k,:),1),nodes(edges(k,:),2),...
        '-','Color',color,'LineWidth',width); end
end




