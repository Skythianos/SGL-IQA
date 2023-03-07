clear all;
close all;
clc;


db_all = { 'mlive' 'mdid2013' };
metric = 'gwhglbp';


%%%% SVR PARAMETERS ON MLIVE DATABASE
bestp = 1;   %%% bestp is one para that depends on DMOS range. For all databases, we scale the dmos into range to [0 100],
%%% and fix the p para as 1.
bestc = 128;  bestg = 1;   %%% SVR PARAS ON MLIVE DB. These two paras are fixed for each db.
for x = 1:1
    
    
    features = [];
    db = db_all{x};
    load( sprintf( '%s_info', db ) );
    load( sprintf( '%s-%s.mat',metric,db));
    
    hist_fv = sqrt(hist_fv);    %%% normalize features
    
    features = [features hist_fv];
    
    if size(mos,1) == 1;
        mos=mos';
    end
    
    
    cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p ',num2str(bestp)];
    
    rts = 1000;
    
    results = zeros(rts, 4);
    noise_num = max( noise_idx(:) );
    results_ind = zeros(rts,noise_num);
    for rt=1:rts
        type_idx = randperm( max(org_label(:)) );
        org_label_idx = type_idx(org_label);
        
        train_num = round(max(org_label(:))*0.8);
        train_data = features( org_label_idx<=train_num,:);
        train_mos = mos( org_label_idx<=train_num );
        
        test_data = features(org_label_idx>train_num,:);
        test_mos = mos( org_label_idx>train_num );
        test_noise_idx = noise_idx( org_label_idx>train_num );
        
        
        train_scale = train_data; test_scale = test_data;
        svr_model = svmtrain(train_mos, train_scale, cmd);
        [pred_mos, accuracy, prob_esti] = svmpredict(test_mos, test_scale, svr_model);
        
        srcc = IQAPerformance(pred_mos(:),test_mos(:),'s');
        krcc = corr(pred_mos(:),test_mos(:),'type','Kendall');
        cc = IQAPerformance(pred_mos(:),test_mos(:),'p');
        rmse = IQAPerformance(pred_mos(:),test_mos(:),'e');
        
        
        results(rt,1) = srcc; results(rt,2) = krcc;
        results(rt,3) = cc; results(rt,4) = rmse;
        
        
        for nidx = 1 : noise_num
            test_mos_c = test_mos( test_noise_idx==nidx );
            pred_mos_c = pred_mos( test_noise_idx==nidx );
            
            srcc = IQAPerformance(pred_mos_c(:),test_mos_c(:),'s');
            
            results_ind(rt,nidx) = [srcc];
        end
        
        clc;
        
    end
    frlist =  median(results);    %% overall performance on whole database
    frlist_ind = median(results_ind);     %%% SRCC performance on individual distortion types
    
end
