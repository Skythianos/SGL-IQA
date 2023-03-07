function [output] = getFeatures(img)

    YCbCr = rgb2ycbcr(img);
       
    f1 = Grad_LOG_CP_TIP(rgb2gray(img));
    f2 = feature_extract(rgb2gray(img));
    f3 = feature_extract_SSEQ(img,3);
    f4 = gwhglbp_feature(rgb2gray(img));
    f5 = feature_extract_curvelet(img);
    f6 = brisque_feature(im2double(rgb2gray(img)));
    f7 = higrade_1(img);
    f8 = Derivative(img);
    f9 = Bilaplacian(img);
    f10= [HighBoost(YCbCr(:,:,1)), HighBoost(YCbCr(:,:,2)), HighBoost(YCbCr(:,:,3))];
    f11= extractHOGFeatures(YCbCr(:,:,1));
    f12= extractHOGFeatures(YCbCr(:,:,2));
    f13= extractHOGFeatures(YCbCr(:,:,3));

    F1 = [mean(f1), median(f1), std(f1), kurtosis(f1), skewness(f1)];
    F2 = [mean(f2), median(f2), std(f2), kurtosis(f2), skewness(f2)];
    F3 = [mean(f3), median(f3), std(f3), kurtosis(f3), skewness(f3)];
    F4 = [mean(f4), median(f4), std(f4), kurtosis(f4), skewness(f4)];
    F5 = [mean(f5), median(f5), std(f5), kurtosis(f5), skewness(f5)];
    F6 = [mean(f6), median(f6), std(f6), kurtosis(f6), skewness(f6)];
    F7 = [mean(f7), median(f7), std(f7), kurtosis(f7), skewness(f7)];
    F8 = [mean(f8), median(f8), std(f8), kurtosis(f8), skewness(f8)];
    F9 = [mean(f9), median(f9), std(f9), kurtosis(f9), skewness(f9)];
    F10= [mean(f10), median(f10), std(f10), kurtosis(f10), skewness(f10)];
    F11= [mean(f11), median(f11), std(f11), kurtosis(f11), skewness(f11)];
    F12= [mean(f12), median(f12), std(f12), kurtosis(f12), skewness(f12)];
    F13= [mean(f13), median(f13), std(f13), kurtosis(f13), skewness(f13)];

    output = [F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13];

    output(isnan(output))=0;
    output(isinf(output))=0;

end

function out = Grad_LOG_CP_TIP(imd)

sigma = 0.5;
[gx,gy] = gaussian_derivative(imd,sigma);
grad_im = sqrt(gx.^2+gy.^2);

window2 = fspecial('log', 2*ceil(3*sigma)+1, sigma);
window2 =  window2/sum(abs(window2(:)));
log_im = abs(filter2(window2, imd, 'same'));

ratio = 2.5; % default value 2.5 is the average ratio of GM to LOG on LIVE database
grad_im = abs(grad_im/ratio);

%Normalization
c0 = 4*0.05;
sigmaN = 2*sigma;
window1 = fspecial('gaussian',2*ceil(3*sigmaN)+1, sigmaN);
window1 = window1/sum(window1(:));
Nmap = sqrt(filter2(window1,mean(cat(3,grad_im,log_im).^2,3),'same'))+c0;
grad_im = (grad_im)./Nmap;
log_im = (log_im)./Nmap;
% remove the borders, which may be the wrong results of a convolution
% operation
h = ceil(3*sigmaN);
grad_im = abs(grad_im(h:end-h+1,h:end-h+1,:));
log_im = abs(log_im(h:end-h+1,h:end-h+1));

ctrs{1} = 1:10;ctrs{2} = 1:10;
% histogram computation
step1 = 0.20;
step2 = 0.20;
grad_qun = ceil(grad_im/step1);
log_im_qun = ceil(log_im/step2);

N1 = hist3([grad_qun(:),log_im_qun(:)],ctrs);
N1 = N1/sum(N1(:));
NG = sum(N1,2); NL = sum(N1,1);

alpha1 = 0.0001;
% condition probability: Grad conditioned on LOG
cp_GL = N1./(repmat(NL,size(N1,1),1)+alpha1);
cp_GL_H=  sum(cp_GL,2)';
cp_GL_H = cp_GL_H/sum(cp_GL_H);
% condition probability: LOG conditioned on Grad
cp_LG = N1./(repmat(NG,1,size(N1,2))+alpha1);
cp_LG_H = sum(cp_LG,1);
cp_LG_H = cp_LG_H/(sum(cp_LG_H));

out = [NG', NL, cp_GL_H,cp_LG_H];
end


function [gx,gy] = gaussian_derivative(imd,sigma)
window1 = fspecial('gaussian',2*ceil(3*sigma)+1+2, sigma);
winx = window1(2:end-1,2:end-1)-window1(2:end-1,3:end);winx = winx/sum(abs(winx(:)));
winy = window1(2:end-1,2:end-1)-window1(3:end,2:end-1);winy = winy/sum(abs(winy(:)));
gx = filter2(winx,imd,'same');
gy = filter2(winy,imd,'same');
end

function f= feature_extract(im)
%% scale 1:
    [RO, GM, RM]=FGr(im);% compute the RO RM and GM map

    f1=VarInformation(GM, 2);% compute the statistics variance of GM

    f2=VarInformation(RO, 1);% compute the statistics variance of RO
    
    f3=VarInformation(RM, 2);% compute the statistics variance of RM

%% scale 2:
    im2=imresize(im,0.5);
    [RO2, GM2, RM2]=FGr(im2);% compute the RO RM and GM map in size 2

    f4=VarInformation(GM2, 2);% compute the statistics variance of GM in size 2
        
    f5=VarInformation(RO2, 1);% compute the statistics variance of RO in size 2
    
    f6=VarInformation(RM2, 2);% compute the statistics variance of RM in size 2

%% feature
    f=[f1, f2, f3, f4, f5, f6];
end

function V= VarInformation(im, type)
% compute the statistics variance of RO, RM and GM
%   Detailed explanation goes here
% im=double(im);
mintemp=min(im(:))-1;
maxtemp=max(im(:))+1;

if type==1
    x=round(mintemp):0.3:round(maxtemp); %quantization for RO
elseif type==2
    x=round(mintemp):round(maxtemp); %quantization for GM or RM
end
his=hist(im(:),x); 
his=his/sum(his(:)); %do statistics analysis

V=std(his(:)); %compute the variance of statistics
end

function [RO, GM, RM] = FGr(im)
% compute the RO RM and GM map
%   Detailed explanation goes here

sigma = 0.5;
[im_Dx,im_Dy] = gaussian_derivative(im,sigma);% compute the basic gradient maps in the horizontal x and vertical y directions

aveKernel = fspecial('average', 3);
eim_Dx = conv2(im_Dx, aveKernel,'same');
eim_Dy = conv2(im_Dy, aveKernel,'same');% compute the average directional derivative

im_D=atan(eim_Dx./(eim_Dy));
im_D(eim_Dy==0)=pi/2;

RO=atan(im_Dx./(im_Dy));
RO(im_Dy==0)=pi/2;
RO=RO-im_D; % compute RO

GM=sqrt(im_Dx.^2+im_Dy.^2); % compute GM

RM=sqrt((im_Dx-eim_Dx).^2+(im_Dy-eim_Dy).^2); %compute RM

end
