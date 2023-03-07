clear all;
close all;
clc;


db_all = { 'mlive' 'mdid2013' };
metric = 'gwhglbp';

%%%% SVR PARAMETERS ON MLIVE DATABASE
bestp = 1;   %%% bestp is one para that depends on DMOS range. For all databases, we scale the dmos into range to [0 100],
%%% and fix the p para as 1.
bestc = 256;  bestg = 1;   %%% SVR PARAS ON MDID2013 DB.  These two paras are fixed for each db.
for x = 2:2
    
    
    features = [];
    db = db_all{x};
    load( sprintf( '%s_info', db ) );
    load( sprintf( '%s-%s.mat',metric,db));
    
    hist_fv = sqrt(hist_fv);   %%%% feature normalization
    
    features = [features hist_fv];
    
    if size(mos,1) == 1;
        mos=mos';
    end
    
    
    %%% The DMOS range on MDID2013 IS [0 1]. We map it to the DMOS range of
    %%% [0 100] to make all databases consistent.
    %%% Also in this way, we can fix SVR parameter p to be 1 across all
    %%% databases.
    [mos_map,psmap] = mapminmax(mos',0,100);
    mos_map = mos_map';
    
    cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p ',num2str(bestp)];
    
    
    rts = 1000;
    
    
    results = zeros(rts,4);
    
    for rt=1:rts
        type_idx = randperm( max(org_label(:)) );
        org_label_idx = type_idx(org_label);
        
        train_num = round(max(org_label(:))*0.8);
        train_data = features( org_label_idx<=train_num,:);
        train_mos = mos_map( org_label_idx<=train_num );
        test_data = features(org_label_idx>train_num,:);
        test_mos = mos( org_label_idx>train_num );
        test_noise_idx = noise_idx( org_label_idx>train_num );
        
        
        train_scale = train_data; test_scale = test_data;
        svr_model = svmtrain(train_mos, train_scale, cmd);
        [pred_mos, accuracy, prob_esti] = svmpredict(test_mos, test_scale, svr_model);
        
        
        pred_mos_map = mapminmax('reverse',pred_mos',psmap);
        pred_mos = pred_mos_map';
        
        srcc = IQAPerformance(pred_mos(:),test_mos(:),'s');
        krcc = corr(pred_mos(:),test_mos(:),'type','Kendall');
        cc = IQAPerformance(pred_mos(:),test_mos(:),'p');
        rmse = IQAPerformance(pred_mos(:),test_mos(:),'e');
        
        
        results(rt,1) = srcc; results(rt,2) = krcc;
        results(rt,3) = cc; results(rt,4) = rmse;
        
        
    end
    frlist =  median(results);    %% overall performance on whole database
end
