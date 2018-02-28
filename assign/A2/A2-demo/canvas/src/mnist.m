% test load mnist dataset
function data = mnist()
filename='../data/train-images-idx3-ubyte';

% open file
fid =fopen(filename);

% magic number
fread(fid, 1, 'int32', 'b');

% the number of images
nimg=fread(fid, 1, 'int32', 'b');

% the number of rows in every image
nrow=fread(fid, 1, 'int32', 'b');

% the number of cols in every image
ncol=fread(fid, 1, 'int32', 'b');

% load in the data
data=fread(fid, [ncol*nrow, nimg], 'uchar');
fclose(fid);

% convert the row majored data to column majored format
data=permute(reshape(data, [ncol, nrow, nimg]),[2, 1, 3]);
end