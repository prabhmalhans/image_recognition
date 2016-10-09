This Matlab program performs image recognition using backpropagation algorithm in neural network.It recognises 9 categories of images by default that can be changed accoerding to the requirement. The images belonging to various categories were downloaded via a compressed zip file http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
Principle component analysis is used to extract useful features of the image.  

Following .m files are used throughout the program:-

1. Main.m:- It is the main program file that calls all other function .m files. 
	
	It calls read_images.m to read the dataset to be learnt in X and target data in Y.
	X is mxn matrix of m training examples with each examples having n features.
	
	PCA reduces the normalized X to mxK matrix where K<n to reduce complexity of the calculations.

	You can refer to Professor Andrew Ng's lecture on neural networks to understand the backpropagation algorithm.
2.fmincg.m:- Function used to train the cost function.

3.nnCostFunction.m:-Calculates Cost and gradient required by fmincg.m to train the network.

4.testreco.m:-  To perform testing of the image recognition. To be run separately after training completes.
