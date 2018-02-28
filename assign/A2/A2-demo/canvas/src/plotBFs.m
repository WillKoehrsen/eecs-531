function [h,r] = plotBFs(A, layout, scalemode, selmode, uidx)
% plotApatch -- plot the columns of matrix A as image patches
%   Usage
%     plotApatch(A [, layout, bgscale, selmode])
%   Inputs
%      A          basis matrix
%      layout     [nrows, ncols] specifies the layout of the plot.
%                 A scalar value generates an approximately square layout.
%                 If null, layout defaults to the number of columns in A.
%                 Optional third argument specifies integer scaling
%                 of pixel size on screen.
%      scalemode  type of scale:
%                    'each'    scale each vector to max gray range (default)
%                    'same'    use same scale for all plotted vectors
%                    'zeroc'   like each, but with center on zero
%
%      selmode    how to select the basis vectors to plot:
%                    'rand'    plot random unique vectors
%                    'ord'     plot vectors in order
%                    'l2ord'   plot vectors in order of their L2 norm
%                    'sim'     plot vectors in order of similarity
%                    'user     plot vectors in order given by uidx
%                    n         print every nth vector in L2 order
%                 selmode defaults to 'l2ord' if plotting all basis vectors
%                 and 'rand' if plotting fewer.
%      uidx
%   Outputs
%      h          graphics handles of each subplot
%      r          the index of each subplot

[L,M] = size(A);
n = sqrt(L);
if (n ~= round(n)) 
  fprintf('Vector size must be square.\n');
  return;
end

pixelsize = 0;		% don't match bf pixels to screen pixels
spacing   = 1;		% in terms of bf pixels

if nargin < 2 | isempty(layout)
  layout = M;
end
if length(layout) == 1
  nvecs = min(M,layout);
  nrows = ceil(sqrt(nvecs));
  ncols = ceil(nvecs/nrows);
else
  nrows = layout(1);
  ncols = layout(2);

  if length(layout) == 3
    pixelsize = layout(3);
  end

  nvecs = min(M,nrows*ncols);
end

if nargin < 3
  scalemode = 'zeroc';
end

if nargin < 4
  if nvecs < M
    selmode = 'rand';
  else
    selmode = 'l2ord';
  end
end

% make bfs take up a greater pct of fig area
pctBorder = 5;
axOffset = 0.5*pctBorder/100;
axSize = 1 - pctBorder/100;
set(gca,'Position',[axOffset axOffset axSize axSize]);

% tighten up plot, keep screen position same
fpos = get(gcf,'Position');
figwidth = fpos(3);
figheight = fpos(4);

if pixelsize == 0
  s = min(figwidth/ncols,figheight/nrows);
  xsize = ncols*s;
  ysize = nrows*s;

  fxpos = fpos(1);
  fypos = fpos(2) + (figheight - ysize);
  set(gcf,'Position',[fxpos fypos xsize ysize]);

else
  s = 100/(100 - pctBorder);

  xsize = ceil(s*pixelsize*(ncols*sqrt(L) + (ncols+1)*spacing));
  ysize = ceil(s*pixelsize*(nrows*sqrt(L) + (nrows+1)*spacing));

  fxpos = fpos(1);
  fypos = fpos(2) + (figheight - ysize);
  set(gcf,'Position',[fxpos fypos xsize ysize]);
end
 
% change the size of the window for printing square patches
s = min(8/ncols,10.5/nrows);
set(gcf,'PaperPosition',[0.25 0.25 s*ncols s*nrows]);

everyn = 1;
if isnumeric(selmode)
  everyn = selmode;
  selmode = 'l2ord';
end

if strcmp(selmode,'ord')
  pidx = 1:M;
elseif strcmp(selmode,'l2ord')
  nA = zeros(1,M);
  for m=1:M
    nA(m) = norm(A(:,m));
  end
  [nA pidx] = sort(-nA);
elseif strcmp(selmode,'sim')
  [Ar, pidx] = reorderA(A);
elseif strcmp(selmode,'user')
  pidx = uidx;
else
  pidx = randperm(M);
end

r = pidx(1:everyn:M);
r = r(1:nvecs);

amin = min(A(:));
amax = max(A(:));

I = ones(n*nrows + (nrows+1)*spacing, n*ncols + (ncols+1)*spacing);

% set background gray level (-1 = black, 1 = white)
bgscale = -1;
I = I*bgscale;

c = 1;
for px=1:nrows
  for py=1:ncols
    % convert A into an image so we can plot with imagesc
    patch = reshape(A(:,r(c)),n,n);

    % rescale patch to be in range [-1,1]
    if strcmp(scalemode,'each')
      % scale each vector to max gray range
      pmin = min(patch(:));
      pmax = max(patch(:));
      patch = (patch - pmin) / (pmax - pmin);
    elseif strcmp(scalemode,'zeroc')
      % like each, but with center on zero
      pmin = min(patch(:));
      pmax = max(patch(:));
      if (abs(pmax) > abs(pmin))
	patch = patch / abs(pmax);
      else
	patch = patch / abs(pmin);
      end
    elseif strcmp(scalemode,'same')
      patch = (patch - amin) / (amax - amin);
    else
        patch = patch .* 2 - 1;
    end

    ix = (px - 1)*n + px*spacing + 1;
    iy = (py - 1)*n + py*spacing + 1;

    I(ix:ix+n-1,iy:iy+n-1) = patch;

    c = c + 1;
    if c > nvecs
      break
    end
  end
end

h = imagesc(I,[-1,1]);
axis off;
colormap gray;
