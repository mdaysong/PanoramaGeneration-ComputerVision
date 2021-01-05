clear;
clc;

%read image
original = imread('I1.jpg');
addOn = imread('I2.jpg');

%detect, extract features for I_1
I_1 = rgb2gray(original);
pointsI_1 = detectSURFFeatures(I_1);
[featuresI_1,vPointsI_1] = extractFeatures(I_1,pointsI_1);

%detect, extract features for I_2
I_2 = rgb2gray(addOn);
pointsI_2 = detectSURFFeatures(I_2);
[featuresI_2,vPointsI_2] = extractFeatures(I_2,pointsI_2);

%size of I_1 and I_2
[numRowsI_1, numColsI_1] = size(I_1);
[numRowsI_2, numColsI_2] = size(I_2);

%match features, get locations
indexPairs = matchFeatures(featuresI_1,featuresI_2);
%takes location of the matching surfpoint of I_1
matchedPointsI_1 = vPointsI_1(indexPairs(:,1));
matchedPointsI_2 = vPointsI_2(indexPairs(:,2));

%eg: 1st row match with 2nd row
%matchedpointI_1 = (x,y)
%matchedpointI_2 = (x',y')
matchedLocI_1 = matchedPointsI_1.Location;
matchedLocI_2 = matchedPointsI_2.Location;

%matchedI_1Norm is normalized, matchedI_2Norm is normalized
%format: [x x; y y; z z];
[matchedI_1Norm, matchedI_2Norm, M1, M2] = normalise2dpts(matchedLocI_1, matchedLocI_2);

%get the size of a matrix that contains the locations
[numRows, numCols] = size(matchedLocI_1);

%get the resulting H normalized matrix from the ransac function
[Htemp, A, tempo] = tempRansacFun(numRows, matchedI_1Norm, matchedI_2Norm);

%denormalise the H matrix
H = (inv(M2)*Htemp*M1).*(-1);

%get height and widths of the initial images
[heightI2, widthI2] = size(I_2);
[heightI1, widthI1] = size(I_1);

%call getSizeBackground to get the new height, new width and 4 corners of
%the new image
[newHeight, newWidth, newX, newY, xB, yB] = getSizeBackground(H, heightI2, widthI2, heightI1, widthI1);

%create meshgrid with coordinate system using the original dimensions
[X,Y] = meshgrid(1:widthI2,1:heightI2);
%create meshgrid with the newer dimensions from getSizeBackground
[newMeshX,newMeshY] = meshgrid(newX:newX+newWidth-1, newY:newY+newHeight-1);

fullDimension = newHeight*newWidth;
preComposite = ones(3,fullDimension);

%reshape rows of the preComposite matrix based on the mesh x and mesh y
%sizes
preComposite(1,:) = reshape(newMeshX,1,fullDimension);
preComposite(2,:) = reshape(newMeshY,1,fullDimension);
%apply HMatrixation to the preComposite
preComposite = H*preComposite;
newMeshX = reshape(preComposite(1,:)./preComposite(3,:), newHeight, newWidth);
newMeshY = reshape(preComposite(2,:)./preComposite(3,:), newHeight, newWidth);
 
%allBlack sets a zero layer to black pixel value
allBlack = zeros(size(original, 1), size(addOn, 2), 'uint8');

% interpolation, original image onto the compositeImage, setting the red
% color channel
newImage(:,:,1) = interp2(X, Y, double(addOn(:,:,1)), newMeshX, newMeshY);
newImage(:,:,2) = interp2(X, Y, double(allBlack), newMeshX, newMeshY);
newImage(:,:,3) = interp2(X, Y, double(allBlack), newMeshX, newMeshY);

%setting the green and blue channels for the add on second image
[left,right,channel]=size(original);
secondImage=zeros(left, right); 
secondImage(:,:,2)=original(:,:,2);  
secondImage(:,:,3)=original(:,:,3);

%merging the second image onto the composite image
[newImage] = copyImg(newImage, secondImage, xB, yB);
 
%show the image
imshow(uint8(newImage));



%normalization function
function [matchedI_1Norm, matchedI_2Norm, M1, M2] = normalise2dpts(matchedI_1, matchedI_2)

    [numRows, numCols] = size(matchedI_1);
    
    lastCoordinate = ones(1,numRows);

    %normalizing for regular (x,y), perform necessary calculations for mean
    %and standard deviation for the values in image 1
    %obtain individual x and y coordinates
    allXI_1 = matchedI_1(:,1);
    allYI_1 = matchedI_1(:,2);

    %calculate mean and std deviation
    XI_1mean = sum(allXI_1)/numRows;
    YI_1mean = sum(allYI_1)/numRows;
    stdDev1 = sqrt((sum((allXI_1 - XI_1mean).^2 + (allYI_1 - YI_1mean).^2))/(2*numRows));

    %perform normalization
    index = 1:numRows;
    matchedI_1NormX = ((allXI_1 - XI_1mean)/stdDev1)';
    matchedI_1NormY = ((allYI_1 - YI_1mean)/stdDev1)';
    
    %concatenate all array values for the normalised values
    matchedI_1Norm = vertcat(matchedI_1NormX, matchedI_1NormY, lastCoordinate);
    
    %calculate M1 when needing to denormalise H in the future
    M1L = [1/stdDev1 0 0;
            0 1/stdDev1 0;
            0 0 1];
    M1R = [1 0 (-1)*XI_1mean;
            0 1 (-1)*YI_1mean;
            0 0 1];
    
    M1 = M1L * M1R;
        
        
    %normalizing for (x',y'), perform necessary calculations for mean
    %and standard deviation for the values in image 2
    %obtain individual x and y coordinates
    allXI_2 = matchedI_2(:,1);
    allYI_2 = matchedI_2(:,2);
    
    %calculate mean and std deviation
    XI_2mean = sum(allXI_2)/numRows;
    YI_2mean = sum(allYI_2)/numRows;
    stdDev2 = sqrt((sum((allXI_2 - XI_2mean).^2 + (allYI_2 - YI_2mean).^2))/(2*numRows));

    %perform normalization
    matchedI_2NormX = ((allXI_2 - XI_2mean)/stdDev2)';
    matchedI_2NormY = ((allYI_2 - YI_2mean)/stdDev2)';
    
    %concatenate all array values for the normalised values
    matchedI_2Norm = vertcat(matchedI_2NormX, matchedI_2NormY, lastCoordinate);

    %calculate M2 when needing to denormalise H in the future
    M2L = [1/stdDev2 0 0;
            0 1/stdDev2 0;
            0 0 1];
    M2R = [1 0 (-1)*XI_2mean;
            0 1 (-1)*YI_2mean;
            0 0 1];

    M2 = M2L * M2R;
    
    
end

%ransac function
function [H, A, tempo] = tempRansacFun(N, I_1Norm, I_2Norm)

    count = 0;
    %condition that the consensus set must have > 85% of total points
    while count < (0.85 * N) 

        %randomize 4 indices, creates a horizontal matrix
        randInx = randi([1 N],1,4);

        %finds tuples from the first image
        firstTupleI_1 = I_1Norm(:,randInx(:,1));
        tempo = firstTupleI_1;
        secondTupleI_1 = I_1Norm(:,randInx(:,2));
        thirdTupleI_1 = I_1Norm(:,randInx(:,3));
        fourthTupleI_1 = I_1Norm(:,randInx(:,4));

        %finds tuples from the second image
        firstTupleI_2 = I_2Norm(:,randInx(:,1));
        secondTupleI_2 = I_2Norm(:,randInx(:,2));
        thirdTupleI_2 = I_2Norm(:,randInx(:,3));
        fourthTupleI_2 = I_2Norm(:,randInx(:,4));

        %compute the homography model
        %base for the A matrix
        A = zeros(8,9);

        firstRowA = [firstTupleI_1(1) firstTupleI_1(2) 1 0 0 0 (-1)*firstTupleI_2(1)*firstTupleI_1(1) (-1)*firstTupleI_2(1)*firstTupleI_1(2) (-1)*firstTupleI_2(1)];
        secondRowA = [0 0 0 firstTupleI_1(1) firstTupleI_1(2) 1 (-1)*firstTupleI_2(2)*firstTupleI_1(1) (-1)*firstTupleI_2(2)*firstTupleI_1(2) (-1)*firstTupleI_2(2)];

        thirdRowA = [secondTupleI_1(1) secondTupleI_1(2) 1 0 0 0 (-1)*secondTupleI_2(1)*secondTupleI_1(1) (-1)*secondTupleI_2(1)*secondTupleI_1(2) (-1)*secondTupleI_2(1)];
        fourthRowA = [0 0 0 secondTupleI_1(1) secondTupleI_1(2) 1 (-1)*secondTupleI_2(2)*secondTupleI_1(1) (-1)*secondTupleI_2(2)*secondTupleI_1(2) (-1)*secondTupleI_2(2)];

        fifthRowA = [thirdTupleI_1(1) thirdTupleI_1(2) 1 0 0 0 (-1)*thirdTupleI_2(1)*thirdTupleI_1(1) (-1)*thirdTupleI_2(1)*thirdTupleI_1(2) (-1)*thirdTupleI_2(1)];
        sixthRowA = [0 0 0 thirdTupleI_1(1) thirdTupleI_1(2) 1 (-1)*thirdTupleI_2(2)*thirdTupleI_1(1) (-1)*thirdTupleI_2(2)*thirdTupleI_1(2) (-1)*thirdTupleI_2(2)];

        seventhRowA = [fourthTupleI_1(1) fourthTupleI_1(2) 1 0 0 0 (-1)*fourthTupleI_2(1)*fourthTupleI_1(1) (-1)*fourthTupleI_2(1)*fourthTupleI_1(2) (-1)*fourthTupleI_2(1)];
        eighthRowA = [0 0 0 fourthTupleI_1(1) fourthTupleI_1(2) 1 (-1)*fourthTupleI_2(2)*fourthTupleI_1(1) (-1)*fourthTupleI_2(2)*fourthTupleI_1(2) (-1)*fourthTupleI_2(2)];

        %A matrix
        A = cat(1, firstRowA, secondRowA, thirdRowA, fourthRowA, fifthRowA, sixthRowA, seventhRowA, eighthRowA);
        %svd calculation
        [U, S, V] = svd(A);
        %H is equal to the last column in V as it corresponds with the smallest
        %singular value
        H = V(:, end);
        %put the H matrix in its original 3x3 position
        H = transpose(reshape(H,3,3));

        INormExpected = H * I_1Norm(1:3,1:N);

        %horizontal matrices
        IXNorm = INormExpected(1, 1:N);
        IYNorm = INormExpected(2, 1:N);
        IScale = INormExpected(3, 1:N);

        %remove the weighting on wx*, wy*, 1
        INormExpected(1, 1:N) = IXNorm./IScale;
        INormExpected(2, 1:N) = IYNorm./IScale;
        INormExpected(3, 1:N) = IScale./IScale;

        %compare distances for all values
        distanceMatrix = sqrt((INormExpected(1,1:N) - I_2Norm(1,1:N)).^2 + (INormExpected(2,1:N) - I_2Norm(2,1:N)).^2);
        count = 0;
        count = sum(distanceMatrix(:)< 0.55);

    end
        
    %refit the H to all the points in the consensus set
    AForAll = ones(2*N,9);
    iterCount = 0;
    for i = 1:N
        first = [I_1Norm(1,i) I_1Norm(2,i) 1 0 0 0 (-1)*I_2Norm(1,i)*I_1Norm(1,i) (-1)*I_2Norm(1,i)*I_1Norm(2,i) (-1)*I_2Norm(1,i)];
        iterCount = iterCount + 1;
        AForAll(iterCount, :) = first; 
        second = [ 0 0 0 I_1Norm(1,i) I_1Norm(2,i) 1 (-1)*I_2Norm(2,i)*I_1Norm(1,i) (-1)*I_2Norm(2,i)*I_1Norm(2,i) (-1)*I_2Norm(2,i)];
        iterCount = iterCount + 1;
        AForAll(iterCount, :) = second;
    end
    
    [U, S, V] = svd(AForAll);
    %H is equal to the last column in V as it corresponds with the smallest
    %singular value
    H = V(:, end);
    %put the H matrix in its original 3x3 position
    H = transpose(reshape(H,3,3));

end

%getSizeBackground function for finding the background size of the to be
%stitched panorama
function [newHeight, newWidth, x1, y1, x2, y2] = getSizeBackground(HMatrix, h2, w2, h1, w1)
 
    % create new base panorama and then reshape composite matrix
    [X,Y] = meshgrid(1:w2,1:h2);
    preComposite = ones(3,h2*w2);
    preComposite(1,:) = reshape(X,1,h2*w2);
    preComposite(2,:) = reshape(Y,1,h2*w2);

    % compute and top and left corners 
    newpreComposite = HMatrix\preComposite;
    x1 = fix(min([1,min(newpreComposite(1,:)./newpreComposite(3,:))]));
    y1 = fix(min([1,min(newpreComposite(2,:)./newpreComposite(3,:))]));

    %bottom and right corners
    new_right = fix(max([w1,max(newpreComposite(1,:)./newpreComposite(3,:))]));
    new_bottom = fix(max([h1,max(newpreComposite(2,:)./newpreComposite(3,:))]));

    %set up return values for function
    x2 = 2 - x1;
    y2 = 2 - y1;
    newHeight = new_bottom - y1 + 1;
    newWidth = new_right - x1 + 1;

end

%merging the second image onto the original image for the composite photo
function [compositeImage] = copyImg(image1, image2, x, y)

    %condition that null values at image1 is automatically set to 0
    image1(isnan(image1))=0;

    %keep 1st channel for the red image
    redLayer = (image1(:,:,1)>0);
    %redLayer = (image1(:,:,1)>0 |image1(:,:,2)>0 | image1(:,:,3)>0);
    compositeImage = zeros(size(image1));
    
    %replace the appropriate dimensions of the composite value and overlap
    %with the second image
    compositeImage(y:y+size(image2,1)-1, x: x+size(image2,2)-1,:) = image2;
    compositeLayer = (compositeImage(:,:,1)>0 | compositeImage(:,:,2)>0 | compositeImage(:,:,3)>0);
    compositeLayer = and(redLayer, compositeLayer);
    compositeLayer = ones(size(compositeLayer));

    %overlap the intial color channel within both images
     compositeImage(:,:,1) = image1(:,:,1).*compositeLayer + compositeImage(:,:,1);

end
