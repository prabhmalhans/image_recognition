function [X,Y] = read_images()
categories = {'airplanes';'bonsai';'brain';'car_side';'chandelier';'Faces';'hawksbill';'Leopards';'Motorbikes'};
sample_size = 85;
A=[];
for i=1:length(categories)
    for j=1:sample_size
    string=sprintf('F:\\Image recognition project\\101_ObjectCategories\\%s\\image_%1.4d.jpg',cell2mat(categories(i)),0000+j);
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
A = A(randperm(size(A,1)),:);
X=double(A(:,1:(size(A,2)-1)));
Y=double(A(:,size(A,2)));
    
    

   
