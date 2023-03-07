clear,clc
im =imread('image1.bmp');
path(path,[pwd '\fdct_usfft_matlab;']);
fdct_usfft_path;
image_feature = feature_extract(im);
