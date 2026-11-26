% Reviewer-requested sensitivity analyses using the chapter implementation.
% Run from the original project root. Each row is one model and one run.
outdir=fullfile(pwd,'review_revision_analysis'); if ~isfolder(outdir),mkdir(outdir);end
cfg.stageHours=[1 6 12]; hyd=fullfile(pwd,'study_c_data','study_c_hydraulic_responses.mat');
modes=["main","holdline","lowpsi","highpsi","noise","drop","wide","small","large"];
for mode=modes
  nrun=5; if mode=="main", nrun=20; elseif mode=="holdline", nrun=26;end
  rows=[];
  for rr=1:nrun
    c=load_benchmark_system(fullfile(pwd,'benchmark_data'),20260802+rr,hyd,mode=="wide");
    a=c;b=c; nt=160;cfg.measurementNoise=.16; cfg.groupDrop=(mode=="drop");cfg.waterImpactThresholdPsi=1;
    if mode=="holdline"
      a.pumpOutageCases=setdiff(a.pumpOutageCases,rr); a.evOnlyOutageCases=setdiff(a.evOnlyOutageCases,rr);
      b.pumpOutageCases=rr;b.evOnlyOutageCases=rr;
    elseif mode=="noise",cfg.measurementNoise=.32;
    elseif mode=="lowpsi",cfg.waterImpactThresholdPsi=.5;
    elseif mode=="highpsi",cfg.waterImpactThresholdPsi=2;
    elseif mode=="small",nt=80;
    elseif mode=="large",nt=320;end
    sets={make_cascade_dataset(a,nt,23000+rr,.05,cfg),make_cascade_dataset(a,80,24000+rr,.05,cfg),make_cascade_dataset(b,180,25000+rr,.05,cfg)};
    models=[1 3]; if mode=="main",models=1:6;end
    for model=models
      xx=cell(1,3);
      for split=1:3
        ds=sets{split}; xx{split}=ds.views{min(model,3)};
        if model>=4
          x=ds.views{1}; h=ds.views{3}; x(:,16)=h(:,16);
          if model==5
            op=row_normalize(double((c.incidence*c.incidence')>0));
            r=reshape(x(:,4),c.nNodes,[]); mu=op*r; ab=op*abs(r); sd=sqrt(max(op*(r.^2)-mu.^2,0)); coh=abs(mu)./max(ab,1e-8);
            x(:,12:15)=[mu(:) ab(:) sd(:) coh(:)];
          elseif model==6,x(:,12:16)=0;end
          xx{split}=x;
        end
      end
      [xt,xv,xe]=standardize_views(xx{1},xx{2},xx{3});
      tr=sets{1};v=sets{2};t=sets{3}; vm=scenario_node_mask(v,v.scenarioKinds==1); tm=scenario_node_mask(t,t.scenarioKinds==1);
      off=model; if model>=4,off=3;end
      ni=train_mlp(xt,tr.labels,14,140,.008,27000+100*rr+off); nd=train_mlp(xt,tr.dependencyLabels,14,140,.008,37000+100*rr+off);
      pv=predict_mlp(ni,xv);pt=predict_mlp(ni,xe);dv=predict_mlp(nd,xv);dt=predict_mlp(nd,xe); th=select_threshold(v.labels(vm),pv(vm)); td=select_threshold(v.dependencyLabels(vm),dv(vm));
      vals=[];for mask={tm,tm&repmat(c.nodeType==1,180,1),tm&repmat(c.nodeType==2,180,1),tm&repmat(c.nodeType==3,180,1)}
        z=mask{1};m=binary_metrics(t.labels(z),pt(z),th);vals(end+1)=m.f1;
      end
      m=binary_metrics(t.dependencyLabels(tm),dt(tm),td);vals(end+1)=m.f1;
      for k=[1 5 10]
        vt=select_threshold(v.scenarioLabels,aggregate_event_scores(pv,c.nNodes,k));m=binary_metrics(t.scenarioLabels,aggregate_event_scores(pt,c.nNodes,k),vt);vals(end+1)=m.f1;
      end
      if mode=="main" && rr==1 && model==3
        j=find(t.scenarioPumpInvolved & t.scenarioStages==1,1); idx=c.outageCases(t.outageLog(j)); g=idx.groupIndices(find(c.groupKinds(idx.groupIndices)=="pump",1));
        assets=c.groups{g};rrs=t.views{3}((j-1)*c.nNodes+assets,4);
        writematrix([assets(:),rrs(:)],fullfile(outdir,'early_case.csv'));
      end
      rows(end+1,:)=[rr model vals]; 
    end
    writetable(array2table(rows,'VariableNames',{'Run','Model','All','Power','Water','EV','Dependency','K1','K5','K10'}),fullfile(outdir,mode+".csv"));
  end
end

function city=load_benchmark_system(dataDir,shuffleSeed,hydraulicFile,wide)
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
    if isempty(groupIndices)||(~wide && (numel(groupIndices)>6||numel(disconnected)>12)), continue; end
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
outageLog=zeros(nScenarios,1);
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
    missing=rand(n,1)<dropRate; if cfg.groupDrop
        gg=randi(numel(city.groups)); missing(city.groups{gg})=true;
    end
    measurement(missing)=0; rate(missing)=0;
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
    
    if scenarioKind, outageLog(t)=outageIndex; end
    labels(rows)=y; dependencyLabels(rows)=dependencyY;
end
out.outageLog=outageLog; out.views=views; out.labels=labels; out.dependencyLabels=dependencyLabels;
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
    pred=p>=value; tp=sum(pred & y==1); prec=tp/max(sum(pred),1); rec=tp/max(sum(y),1); f=2*prec*rec/max(prec+rec,eps);
    if f>best, best=f; threshold=value; end
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



