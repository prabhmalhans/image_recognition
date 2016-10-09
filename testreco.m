clc 
clear all
load('para.mat');
K=300;
categories = {'airplanes';'bonsai';'brain';'car_side';'chandelier';'Faces';'hawksbill';'Leopards';'Motorbikes'};
test_sample_size = 10;
A=[];
for i=1:length(categories)
    for j=1:test_sample_size
    string=sprintf('F:\\Image recognition project\\101_ObjectCategories\\%s\\image_%1.4d.jpg',cell2mat(categories(i)),0085+j);
    img=imread(string);
    [rows columns numberOfColorChannels] = size(img);
        if numberOfColorChannels > 1
            img = rgb2gray(img);
        else
            img = img; % It's already gray.
        end
    img=imresize(img,[18 28]);
    img=img';
    img=img(:);
    img=img';
    img=[img i];
    A=[A ;img];
    end
end
X=A(:,1:(size(A,2)-1));
Y=A(:,size(A,2))
X=double(X);
Y=double(Y);
X_norm=bsxfun(@minus, X, mu);
X_norm = bsxfun(@rdivide, X_norm, sigma);
X = projectData(X_norm, U, K);
m=size(X);
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y)) * 100);
