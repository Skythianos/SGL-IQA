clear all
close all

addpath(genpath('brisque'));
addpath(genpath('CurveletQA_release'));
addpath(genpath('GRAD_LOG_CP_TIP'));
addpath(genpath('GWH-GLBP-BIQA'));
addpath(genpath('higradeRelease'));
addpath(genpath('OG-IQA_release'));
addpath(genpath('RobustBRISQUE-main'));
addpath(genpath('SSEQ'));

load CLIVE.mat

numberOfSplits = 1;

PLCC = zeros(1,numberOfSplits);
SROCC = zeros(1,numberOfSplits);
KROCC = zeros(1,numberOfSplits);

numberOfImages = size(AllMOS_release,2);
Features = zeros(numberOfImages, 65); 

path = 'C:\Users\User\Downloads\Databases\IQA\ChallengeDB_release\Images';

tic
disp('Feature extraction');
parfor i=1:numberOfImages
    if(mod(i,10)==0)
        disp(i);
    end
    img = imread( strcat(path, filesep, AllImages_release{i}) );
    Features(i,:) = getFeatures(img);
end
toc

F=Features;

disp('Evaluation');
for i=1:numberOfSplits
    rng(i);
    if(mod(i,10)==0)
        disp(i);
    end
    p = randperm(numberOfImages);
    
    Data = F(p,:);
    Target = AllMOS_release(p);
    
    Train = Data(1:round(numberOfImages*0.8),:);
    TrainLabel = Target(1:round(numberOfImages*0.8));
    
    Test  = Data(round(numberOfImages*0.8)+1:end,:);
    TestLabel = Target(round(numberOfImages*0.8)+1:end);
    
    Mdl = fitrgp(Train,TrainLabel','KernelFunction','rationalquadratic','Standardize',true);
    
    Pred = predict(Mdl, Test);
    eval = metric_evaluation(Pred,TestLabel);
    
    PLCC(i) = eval(1);
    SROCC(i)= eval(2);
    KROCC(i)= eval(3);
end

disp('----------------------------------');
X = ['Median PLCC after 100 random train-test splits: ', num2str(round(median(PLCC(:)),3))];
disp(X);
X = ['Median SROCC after 100 random train-test splits: ', num2str(round(median(SROCC(:)),3))];
disp(X);
X = ['Median KROCC after 100 random train-test splits: ', num2str(round(median(KROCC(:)),3))];
disp(X);

rmpath(genpath('brisque'));
rmpath(genpath('CurveletQA_release'));
rmpath(genpath('GRAD_LOG_CP_TIP'));
rmpath(genpath('GWH-GLBP-BIQA'));
rmpath(genpath('higradeRelease'));
rmpath(genpath('OG-IQA_release'));
rmpath(genpath('RobustBRISQUE-main'));
rmpath(genpath('SSEQ'));