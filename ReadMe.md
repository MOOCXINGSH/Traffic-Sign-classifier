# Traffic Sign Classifier
Traffic sign classification using convolutional neural network

**The steps for recognizing Traffic-Sign are the following:**

* Load the data set (see below for links to the project data set)

* Explore, summarize and visualize the data set

* Design, train and test a model architecture

* Use the model to make predictions on new images

* Analyze the softmax probabilities of the new images

* Summarize the results with a written report


### Loading the dataset
* The traffic-sign classifier data is stored in directory called **traffic-signs-data**

* Load the data using pickle module.

* The data comes with three section **training set** , **validation set**  and **test set**. 

* But every data set has four keys and we only need features and labels from each set

* Then I store the training data in X_train and y_train for features and labels respectively.

* Used same technique for validation set and test set also.


### Exploring the dataset and visulazie some features 
For exploring the dataset I used pandas,numpy and collection module and for visualization 
I used matplotlib.pyplot module .
The python inbuilt len() function is used to see the sizes of the datasets.
number of example in training set
number of example in validation set
number of example in test set
Then I used numpy's shape method to get the shape of the data sets as the datasets are in numpy.ndarray class format.
Each dataset has shape of 4-d tensor . The first number is the number of example which I already obtained by len() function.
The last three numbers describing the image shape are height,width and color channel respectively.
All the images are in 32x32x3 shape .
The plot_image() function gives a 4x4 grid of randomly selected 
[image]: ./resource/traffic_sign_images.jpg

Then I print the classes of the labels associated with each 16 images.
The traffic sign data has total 43 classes and each class has different number of examples ranging from approximately 
150 to 2000. We can see the number of examples in each classes by using the Counter() object.
Then I visualized the discrepency in the dataset using a bar plot . Each column gives the number of examples in that class.

### Preprocessing the dataset

**Augmentation of the dataset**
There are many good choices for augmenting the dataset such as adding random contrast or brighness or random flipping by 
left or right . Here I use random brightness to augment the data . I use tensorflow's tf.image.random_brightness() function 
and then evaluate it . 
Then I concat the this new dataset with training data. So the new training size is double than the previous . 
Some of the new images after adding distortions :
[image]: ./resource/distorted_images.jpg

There are many different ways to preprocess the data but I found that mean centred with standard deviation of 1 is the one of
good choice to normalize the data . I used all the color channels. By using the normalizer() function I get the desired result .
Then I shuffled all the training data using scikit-learn's shuffle() function.

### Defining the model for training
I used a model architecture :
                                         
                                         ----------------------------------
                                         |        softmax layer [43] 
                                         ----------------------------------
                                                         ||
                                         ----------------------------------
                                         |   fully connected layer[120]   |
                                         ----------------------------------
                                                         ||
                                         ----------------------------------
                                         |   fully connected layer [400]  |
                                         ----------------------------------
                                                         ||
                                         ----------------------------------
                                         |        dropout[0.4]            |
                                         |    max_pool[2x2/1][valid]      |
                                         | convolution[5x5x256/1][valid]  |
                                         ----------------------------------
                                                         ||
                                         ----------------------------------
                                         |    max_pool[2x2/1][valid]      |
                                         | convolution[5x5x128/1][valid]  |
                                         ----------------------------------
                                                         ||
                                         ----------------------------------
                                         |    max_pool[2x2/1][valid]      |
                                         |  convolution[3x3x32/1][valid]  | 
                                         ----------------------------------
                                                         ||
                                         ----------------------------------
                                         |         input[32x32x3]         |
                                         ----------------------------------
                                         
  
 
 