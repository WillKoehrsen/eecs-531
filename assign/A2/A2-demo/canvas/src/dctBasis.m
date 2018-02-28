function B = dctBasis(u, v, M, N)
% The constant coeffient
cu = sqrt(2.0/M);
if u==0
  cu = 1/sqrt(M);
end

cv = sqrt(2.0/N); 
if v==0
  cv = 1/sqrt(N);
end

% generate the I, J map
[J, I] = meshgrid(0:N-1, 0:M-1);

% compute the basis
T1 = cos( pi .* (u/2.0/M) .* (2*I+1) );
T2 = cos( pi .* (v/2.0/N) .* (2*J+1) ) ;
B = cu .* cv .* T1 .* T2;
end


