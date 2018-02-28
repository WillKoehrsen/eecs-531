% this script is to plot the 64 basic functions of DCT
M=8; 
N=8;
k = 1; % the id of basis
Adct = zeros(M*N, M*N);
for u = 0:M-1
    for v = 0:N-1
        B = dctBasis(u, v, M, N);  
        Adct(k, :) = B(:);
        k = k+1;
    end
end

% plot basic functions
plotBFs(Adct, [M, N], 'zeroc', 'ord');


% save plot
print('../fig/dct-basis.eps', '-depsc')