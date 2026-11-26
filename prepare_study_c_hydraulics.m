%% Precompute EPANET Net3 pressure trajectories for Study C pump outages
clear; close all; clc;
restoredefaultpath;

rootDir=pwd;
dataDir=fullfile(rootDir,'benchmark_data');
outDir=fullfile(rootDir,'study_c_data');
if ~exist(outDir,'dir'), mkdir(outDir); end
epanetRoot=getenv('EPANET_MATLAB_TOOLKIT');
assert(isfolder(epanetRoot),'EPANET-Matlab Toolkit was not found.');
addpath(epanetRoot);

% Rows are: normal, pump 1 failed, pump 2 failed, both pumps failed.
closedPatterns=[0 0;1 0;0 1;1 1];
patternNames={'Normal','Pump 1 failed','Pump 2 failed','Both pumps failed'};
pressure=[]; pumpStatus=[]; pumpNames=[];
originalDir=pwd; cd(dataDir);
for q=1:size(closedPatterns,1)
    d=epanet('Net3.inp');
    pumpIndex=d.getLinkPumpIndex;
    if isempty(pumpNames), pumpNames=string(d.getLinkPumpNameID)'; end
    d.openHydraulicAnalysis; d.initializeHydraulicAnalysis(0);
    tstep=1; T=[]; P=[]; S=[];
    while tstep>0
        for p=1:numel(pumpIndex)
            if closedPatterns(q,p), d.setLinkStatus(pumpIndex(p),0); end
        end
        t=d.runHydraulicAnalysis;
        T(end+1,1)=t; P(end+1,:)=d.getNodePressure;
        S(end+1,:)=d.getLinkStatus(pumpIndex);
        tstep=d.nextHydraulicAnalysisStep;
    end
    d.closeHydraulicAnalysis; d.unload;
    regular=abs(T/3600-round(T/3600))<1e-9&T<24*3600;
    hours=round(T(regular)/3600)+1;
    if isempty(pressure)
        pressure=zeros(size(closedPatterns,1),24,size(P,2));
        pumpStatus=zeros(size(closedPatterns,1),24,numel(pumpIndex));
    end
    pressure(q,hours,:)=P(regular,:);
    pumpStatus(q,hours,:)=S(regular,:);
    fprintf('  hydraulic pattern %d/%d complete\n',q,size(closedPatterns,1));
end
cd(originalDir);

meanAbsPressureChange=zeros(size(closedPatterns,1),1);
maxAbsPressureChange=zeros(size(closedPatterns,1),1);
nodesAboveOnePsi=zeros(size(closedPatterns,1),1);
baseline=squeeze(pressure(1,:,:));
for q=1:size(closedPatterns,1)
    delta=squeeze(pressure(q,:,:))-baseline;
    meanAbsPressureChange(q)=mean(abs(delta),'all');
    maxAbsPressureChange(q)=max(abs(delta),[],'all');
    nodesAboveOnePsi(q)=mean(sum(abs(delta)>=1,2));
end
summaryTable=table(string(patternNames'),meanAbsPressureChange,...
    maxAbsPressureChange,nodesAboveOnePsi,...
    'VariableNames',{'Pattern','MeanAbsolutePressureChangePsi',...
    'MaximumAbsolutePressureChangePsi','MeanNodesAtOrAboveOnePsi'});
writetable(summaryTable,fullfile(outDir,'study_c_hydraulic_summary.csv'));
save(fullfile(outDir,'study_c_hydraulic_responses.mat'),'pressure',...
    'pumpStatus','closedPatterns','patternNames','pumpNames');
disp(summaryTable);
