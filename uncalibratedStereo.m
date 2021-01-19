clear;
clc;

%read image
I1 = imread('left.jpg');
I2 = imread('right.jpg');
%set the images to grayscale
I1 = rgb2gray(I1);
I2 = rgb2gray(I2);

[numRowsI1, numColsI1] = size(I1);
[numRowsI2, numColsI2] = size(I2);

%picture showing grayscale versions of left and right images
figure;
imshow([I1, I2]);

%detect SURF Features
pointsI1 = detectSURFFeatures(I1);
pointsI2 = detectSURFFeatures(I2);

%extract features
[featuresI1,valid_pointsI1] = extractFeatures(I1,pointsI1);
[featuresI2,valid_pointsI2] = extractFeatures(I2,pointsI2);

%match features
%get locations
indexPairs = matchFeatures(featuresI1,featuresI2);
matchedPoints1 = valid_pointsI1(indexPairs(:,1));
matchedPoints2 = valid_pointsI2(indexPairs(:,2));

%compute fundamental matrix
%1 = true, 0 = false
[fRANSAC, inliersIndex] = estimateFundamentalMatrix(matchedPoints1,matchedPoints2,'Method','RANSAC','NumTrials',2000,'DistanceThreshold',1e-4);

%get array of matched point locations
matchLoc1 = matchedPoints1.Location;
matchLoc2 = matchedPoints2.Location;

%obtain only the inlier points
index = find(inliersIndex==1);
mShort1 = [0 0];
mShort1 = vertcat(mShort1, matchLoc1(index,:));
mShort1(1,:) = [];
mShort2 = [0 0];
mShort2 = vertcat(mShort2, matchLoc2(index,:));
mShort2(1,:) = [];

%check the epipole positioning in the image
[isInImage1, epipoleI1] = isEpipoleInImage(fRANSAC, size(I1));
[isInImage2, epipoleI2] = isEpipoleInImage(fRANSAC', size(I1));

%round the epipole values 
epipoleI1X = round(epipoleI1(1));
epipoleI1Y = round(epipoleI1(2));
epipoleI2X = round(epipoleI2(1));
epipoleI2Y = round(epipoleI2(2));

%code to show the epipolar line figures
figure; 
subplot(121);
imshow(I1); 
title("epipole in Image1 is (x,y) = (" + epipoleI1X + ", " + epipoleI1Y + ")"); hold on;
plot(mShort1(:,1),mShort1(:,2),'go', 'MarkerSize', 10)
epiLines = epipolarLine(fRANSAC',mShort2(:,:));
points = lineToBorderPoints(epiLines,size(I1));
line(points(:,[1,3])',points(:,[2,4])');
[isIn1,epipole1] = isEpipoleInImage(fRANSAC,size(I1));

subplot(122);
imshow(I2);
title("epipole in Image2 is (x,y) = (" + epipoleI2X + ", " + epipoleI2Y + ")"); hold on;
plot(mShort2(:,1),mShort2(:,2),'go', 'MarkerSize', 10)
epiLines2 = epipolarLine(fRANSAC',mShort1(:,:));
points2 = lineToBorderPoints(epiLines2,size(I2));
line(points2(:,[1,3])',points2(:,[2,4])');
[isIn2,epipole2] = isEpipoleInImage(fRANSAC,size(I2));

%Todo: Confirm proper matching inlier points (such points need to lie on the epipolar lines)

%estimating uncalibrated rectification
[T1,T2] = estimateUncalibratedRectification(fRANSAC,mShort1,mShort2,size(I2));
%rectify the stereo images
[I1Rect,I2Rect] = rectifyStereoImages(I1,I2,T1,T2);

%parameters necessary for disparity graph
disparityRange = [-56 56];
disparityMap = disparitySGM(I1Rect,I2Rect,'DisparityRange',disparityRange,'UniquenessThreshold',15);

%code for showing the disparity graph
figure;
imshow(disparityMap,disparityRange);
hold on;
title('Disparity Map');
colormap jet;
colorbar







