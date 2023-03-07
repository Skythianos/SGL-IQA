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
