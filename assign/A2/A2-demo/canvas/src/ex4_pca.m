%% EECS 531: Computer Vision
%% A2-Ex4 Principal Components Analysis
% This notebook will show how to learning the PCA basis on the MNIST dataset. 
% 
% The MNIST dataset comprises 60,000 training examples and 10,000 test examples 
% of the handwritten digits 0?9, formatted as 28x28-pixel monochrome images. For 
% more details, please refer to: http://yann.lecun.com/exdb/mnist/

addpath('../src');

%Load all images in mnist to a 3D array with size [height width #images]
data=mat2gray(mnist(), [0, 255]);

% select some samples and plot out
samples = data(:, :, randi(size(data, 3), [1, 64]));
plotBFs(reshape(samples, [], 64), [8 8],  'none', 'ord');
title('Digits in MNIST dataset')
%% PCA analysis

% unroll image to row vectors 
[h, w, n] = size(data);
X = reshape(double(data), h*w, n); 
X = X';

% subtract mean
mu = mean(X);
X = X - repmat(mu,[n, 1]);

% pca
% score = x * coeff
% x is row vector and each column of coeff is a component
[coeff,score,latent] = pca(X); 
%% Plot the "eigen spectrum"

figure
plot(latent);
xlim([1, 361]);
xlabel('the index of eigenvalues')
ylabel('eigenvalues');
grid minor;

% plot the first 50 eigenvalues
figure
plot(latent(1:50));
xlim([1, 50]);
xlabel('the index of eigenvalues')
ylabel('eigenvalues');
grid minor;

%plot the cummulative load
figure
plot(cumsum(latent)./sum(latent));
xlim([1, 361]);
xlabel('number of component');
ylabel('cumulative contribution')
grid minor;
%% Show the first 7 components

figure;
components = reshape(coeff(:, 1:7), [h w 7]);
subplot(2, 4, 1); imshow(reshape(mu, [h, w]), []); title('mean digit')
for k = 1 : 7
    subplot(2, 4, k+1); imshow(components(:, :, k), []); 
    title(['the ' num2str(k) 'th eigenface']);
end
%% Show reconstructed data

id = 255; 
org = data(:, :, id);
mse = zeros(1, 4);
figure;
subplot(2, 5, 1); imshow(org, [0 1]); title('orignial')
for k = 1:4
    rec_x = score(id, 1:k*25) * (coeff(:, 1:k*25)');
    rec = reshape(rec_x+mu, [h, w]);
    err = rec-org;
    mse(k) = mean(err(:).^2); 
    subplot(2, 5,k+1); imshow(rec, [0 1]); title(['k=' num2str(k*25)]);
    subplot(2, 5,k+6); imshow(abs(err), [0 2]); title(['mse=' num2str(mse(k))]);
end
figure
plot([1:4] * 25, mse, '-or', 'LineWidth', 2, 'MarkerEdgeColor', 'k');
xlabel('the number of components')
ylabel('MSE')
grid on
%% Animate the reconstruction process

id = 255;
org = data(:, :, id);
figure;
subplot(1, 2, 1); colormap('gray'); imagesc(org);
subplot(1, 2, 2);
colormap('gray');
hi = imagesc(reshape(mu, [h, w]), [0 1]); 
for k=1:100
    rec_x = score(id, 1:k) * (coeff(:, 1:k)');
    rec = reshape(rec_x+mu, [h, w]);
    set(hi, 'CData', rec);
    drawnow
    pause(0.5)
end