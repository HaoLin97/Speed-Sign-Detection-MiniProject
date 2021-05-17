clear all;
close all;

% If want to change model change the commented line on line 55

% image_dir = '20Kph';
% image_dir = '30Kph';
% image_dir = '50Kph';
% image_dir = '80Kph';
% image_dir = '100Kph';
image_dir = 'All';

% Setting up the standard hog vector
img = imread('GoldStandards V2(1)\670V20.jpg');
img = resize(img);
img = extract_sign(img);

cell_size = 4;

[featureVector_2,hogVisualization_2] = extractHOGFeatures(img, 'CellSize',[cell_size cell_size]);
hogFeatureSize = length(featureVector_2);

% Shuffling and splitting the dataset into training and testing
All_data = imageDatastore(image_dir,'IncludeSubfolders',true,'LabelSource','foldernames');
All_data = shuffle(All_data);
[trainingSet,testSet] = splitEachLabel(All_data, 0.8);

% Use countEachLabel to tabulate the number of images associated with each label
countEachLabel(trainingSet)
% disp(trainingSet.Labels);

% Counting the number of training images
numImages = numel(trainingSet.Files);
trainingFeatures = zeros(numImages,hogFeatureSize,'single');
fullFileNames = vertcat(trainingSet.Files);

% Getting the hog vector of each image
for i = 1:numImages
    img = readimage(trainingSet,i);
    img = resize(img);
%     disp(fullFileNames(i));
    try
        img = extract_sign(img);
%         img = imbinarize(img);
        trainingFeatures(i, :) = extractHOGFeatures(img,'CellSize',[4 4]);
    catch
        disp(fullFileNames(i));
    end
end

% Get labels for each image.
trainingLabels = trainingSet.Labels;

% Fit the training data and labels to the SVM/KNN
classifier = fitcecoc(trainingFeatures, trainingLabels);
% classifier = fitcknn(trainingFeatures, trainingLabels,'NumNeighbors',5,'Standardize',1);
disp('Finished creating model');

% Extracting the hog features for the test dataset, same procedure as that
% of train
testLabels = testSet.Labels;
numImages = numel(testSet.Files);
testFeatures  = zeros(numImages,hogFeatureSize,'single');
fullFileNames = vertcat(testSet.Files);

% Process each image and extract features
for j = 1:numImages
    img = readimage(testSet,j);
    img = resize(img);

    try
        img = extract_sign(img);        
        testFeatures(j, :) = extractHOGFeatures(img,'CellSize', [4 4]);
    catch
        disp(fullFileNames(j));
    end
    
end

% Make class predictions using the test features.
predictedLabels = predict(classifier, testFeatures);

% Tabulate the results using a confusion matrix.
figure();
confusionchart(testLabels, predictedLabels);
disp('Finished Development dataset')

% Evaluating using the stress dataset
testSet = imageDatastore('stress dataset','IncludeSubfolders',true);

numImages = numel(testSet.Files);
testFeatures  = zeros(numImages,hogFeatureSize,'single');
fullFileNames = vertcat(testSet.Files);
% Defining the labels
testLabels = ['40Kph';'40Kph';'30Kph';'30Kph';'20Kph';'20Kph';'20Kph';'40Kph';'40Kph';'40Kph';'40Kph';'40Kph';'40Kph';'40Kph';'40Kph';'20Kph'];
testLabels = categorical(cellstr(testLabels));

% Process each image and extract features
for j = 1:numImages
    img = readimage(testSet,j);
    img = resize(img);
    try
        img = extract_sign(img);
        testFeatures(j, :) = extractHOGFeatures(img,'CellSize', [4 4]);
    catch
        disp(fullFileNames(j));
    end
    
end

% Make class predictions using the test features.
predictedLabels = predict(classifier, testFeatures);
% Plotting onto confusion matrix
figure();
confusionchart(testLabels, predictedLabels);
disp('Finished Stress dataset')


function adjusted = preprocess(image)
% The first type of preporcessing implemented, it is the first one used and
% when this one fails the other preprocess is tried. 
% This method adjusts the contrast of the image and adjusts the
% brightness(intensity) of the image if necessary.

% Balance image such that the historgram of the output has 64 bins and is
% approx. flat.
image = histeq(image);

% Converting the rgb image to gray
greyimage = rgb2gray(image);

% Adjusts the image intensity if the image is overall dark
if mean(greyimage(:)) < 120
 greyimage = imadjust(greyimage);
end
% Returning the processed image
adjusted = greyimage;
end

function adjusted = preprocess_2(rgbImage)
% Second preprocessing method, performs adaptive histogram equalisation on
% the images on the hsv spectrum.
% Later noted that it will reduce performance of a lot of images so only
% used when other approaches fail.

% Converting the rgb image to hsv
hsvImage = rgb2hsv(rgbImage);

% Extract individual color channels.
hChannel = hsvImage(:, :, 1);
sChannel = hsvImage(:, :, 2);
vChannel = hsvImage(:, :, 3);

% Adjust the contrast in the image
vChannel = adapthisteq(vChannel);

% Then, after that recombine the new v channel
% with the old, original h and s channels.
hsvImage2 = cat(3, hChannel, sChannel, vChannel);

% Convert back to rgb.
rgbImage2 = hsv2rgb(hsvImage2);
% Returning the output and also converting it back to uint8
adjusted = uint8(255 * mat2gray(rgbImage2));
end

function extracted = extract_sign(image)
% This methid extracts the sign from the image, which is assumed to be the
% largest circle in the image, only half of the circle is returned as the
% other half is always 0, hence extra useless information 

% Defining the min and max size of the circles to look for
r_min = 10;
r_max = 200;

original_img = image;

% Preprocessing the image and using imfindcircles to find the center and
% radius of the circle
image = preprocess(original_img);
[centers, radii, ~] = imfindcircles(image,[r_min, r_max]);

% If no circles are found, retry with the original image
if isempty(centers)
    [centers, radii, ~] = imfindcircles(original_img,[r_min, r_max]);
    image = rgb2gray(original_img);
    
%   If no circles found again, then use the second preprocess method and
%   look again
    if isempty(centers)
        image = preprocess_2(original_img);
        [centers, radii, ~] = imfindcircles(image,[r_min, r_max]);
    end
    
%     Notifies that no circle found
    if isempty(centers)
        disp('No Circle Found');
    end
end

% Retain the strongest circles according to the metric values.
[~, idx] = max(radii);

% Getting the center and radius of largest circle
largest_centers = centers(idx, :);
largest_radii = radii(idx);

% Getting image size
imagesize = size(image); 
% Creating a mask to be used for isolating the area of interest(cropping)
[xx,yy] = ndgrid((1:imagesize(1))-largest_centers(2),(1:imagesize(2))-largest_centers(1));
mask = uint8((xx.^2 + yy.^2)<(largest_radii^2));

% Cropping the image, but the image size is still the same as original,
% background is black
croppedImage = uint8(zeros(size(image)));
croppedImage(:,:,1) = image(:,:,1).*mask;

% Specifying the size of the rectangle to crop out of image, only the left
% half of image
xmin = largest_centers(1)-largest_radii;
ymin = largest_centers(2)-largest_radii;
width = largest_radii;
height = 2*largest_radii;
rect = [xmin ymin width height];

% Cropping the area of interest, removing the background
cropped_img = imcrop(croppedImage,rect);
% Resizing the image, so that the output is standarised 
cropped_img = imresize(cropped_img, [50,25]);
% Returning the result
extracted = cropped_img;
end

function resized = resize(image)
% Resizes the image accordingly, without losing the shape of the image
[x, y, z] = size(image);
% Doubles the size of image if its too small
while x < 100 || y < 100
    image = imresize(image, [round(x*2), round(y*2)]);
    [x, y, z] = size(image);
end
% Halfs the size of image if its too big
while x > 300 || y > 300
    image = imresize(image, [round(x/2), round(y/2)]);
    [x, y, z] = size(image);
end
resized = image;
end

function blur_meas = calc_blur(image)
% Method implemented to calculate the blurriness of the image, the mehtod
% uses the Laplacian of an image using the imgradient, which returns
% magnitude and angle, then finding the median of the sorted laplace is the
% metric often used to define blurriness.

% Method not used because was not able to implement a good deblur mechanism
gray_image = rgb2gray(image);

[laplace_img,~] = imgradient(gray_image);

nPx = round(0.001*numel(image));

sorted_laplace = sort(laplace_img(:));

measurement = median(sorted_laplace(end-nPx+1:end));

blur_meas = measurement;
end

function deblurred = deblur(image)
% An method that uses the Wiener filter to deblur the image

% The performance was poor, as the variance in the image is too large for a
% fixed filter, but did not get to implement a flexible and effective
% filter.

% Method not used.
I = im2double(image);
LEN = 21;
THETA = 11;
PSF = fspecial('motion', LEN, THETA);
noise_mean = 0;
noise_var = 0.01;
estimated_nsr = noise_var / var(I(:));
wnr3 = deconvwnr(image, PSF, 0.5);

deblurred = wnr3;
end