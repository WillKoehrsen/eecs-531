% -  Read an image and convert the image to grayscale.

orgI = mat2gray( imread('../data/Boats.png') );
imshow(orgI);
%% 

J = dct2(orgI);
figure
imshow(log(abs(J)),[])
colormap(gca,jet(64))
colorbar
%% 

lowpassJ = J;
lowpassJ(40:end, 40:end) = 0;
%% 
% - Reconstruct the image using the inverse DCT function |idct2.|

lowpassI= idct2(lowpassJ);
diffI=abs(orgI-lowpassI);
%% 
 

figure
subplot(1, 3, 1); imshow(orgI); title('original grayscale image');
subplot(1, 3, 2); imshow(lowpassI); title('low pass filtered image');
subplot(1, 3, 3); imshow(diffI); title('absolute difference');
%% 

figure
pind={201:264, 151:214};
subplot(1, 3, 1); imshow(orgI(pind{:})); title('original grayscale image');
subplot(1, 3, 2); imshow(lowpassI(pind{:})); title('low pass filtered image');
subplot(1, 3, 3); imshow(diffI(pind{:})); title('absolute difference');
%% 
% 
% 

highpassJ = J;
highpassJ(1:10, 1:10) = 0;
%% 


figure
highpassI= idct2(highpassJ);
diffI=orgI-highpassI;
figure
subplot(1, 3, 1); imshow(orgI); title('original grayscale image');
subplot(1, 3, 2); imshow(highpassI); title('high pass filtered image');
subplot(1, 3, 3); imshow(diffI); title('absolute difference');
figure
subplot(1, 3, 1); imshow(orgI(pind{:})); title('original grayscale image');
subplot(1, 3, 2); imshow(highpassI(pind{:})); title('high pass filtered image');
subplot(1, 3, 3); imshow(diffI(pind{:})); title('absolute difference');