function f = feature_extract_curvelet( imdist)
    a=rgb2gray(imdist);

%% cut the image into 256*256 image patch.
    a=size8cut(a);
    y=preprocess5(a);

%% deal with the image patch
    [m n]=size(y);
    ff=zeros(1,12);

    for mm=1:m
        for nn=1:n
%% extract feature from one patch
            if(isempty(y{m,n}))
                break;
            else
                c=fdct_usfft(y{mm,nn},1); %curvelet transform
                f1=gaufit_n1(c); %gauss fit the coefficient
                f2=ori_info3(c); %get the Orientational energy feature
                f=[f1,f2];       
                ff=ff+f;
            end
        end
    end
    f=ff/(m*n);              %get the mean of these features in different patch
end

