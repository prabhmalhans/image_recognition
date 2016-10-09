clc
clear all
fprintf('Loading images data...')
[X,Y]=read_images();
[X_norm, mu, sigma] = featureNormalize(X);
[U, S] = pca(X_norm);
K = 300;
X = projectData(X_norm, U, K);

input_layer_size  = 300;  % 300 because of PCA
hidden_layer_size = 100;   % 100 hidden units
num_labels = 9;          % 9 labels, from 1 to 9

Theta1 = randInitializeweights(input_layer_size,hidden_layer_size);
Theta2 = randInitializeweights(hidden_layer_size,num_labels);

nn_params = [Theta1(:) ; Theta2(:)];

fprintf('\nBackpropagation using neural network ...\n')


fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter',150);
lambda = 7
;

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, Y, lambda);
[nn_params, cost] = fmincg(costFunction, nn_params, options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y)) * 100);

save('para.mat','Theta1','Theta2','U','mu','sigma')
