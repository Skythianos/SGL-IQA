function feat=computefeatures(databasepath)
% Class 1 - JP2k
% Class 2 - JPEG
% Class 3 - WN
% Class 4 - Blur
% Class 5 - FF


scalenum = 2;


    imdist = double(rgb2gray(databasepath));     
     

    window = fspecial('gaussian',7,7/6);
    window = window/sum(sum(window));

    feat = [];

    for itr_scale = 1:scalenum

        mu            = filter2(window, imdist, 'same');
        mu_sq         = mu.*mu;
        sigma         = sqrt(abs(filter2(window, imdist.*imdist, 'same') - mu_sq));
        structdis     = (imdist-mu)./(sigma+1);
        L             = lmom(structdis(:));

        feat          = [feat L(2) L(4)];
        shifts        = [ 0 1;1 0 ; 1 1; -1 1];
 
        for itr_shift =1:4
 
            shifted_structdis        = circshift(structdis,shifts(itr_shift,:));
            pair                     = structdis(:).*shifted_structdis(:);
            L                        = lmom(pair(:));
            feat                     = [feat L(1) L(2) L(3) L(4)];

        end


        imdist      = imresize(imdist,0.5);

    end




