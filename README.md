## Background

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. 
With your knowledge of machine learning and neural networks,
you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

## Step 1: Preprocess the Data

Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.
Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.
Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
What variable(s) are the target(s) for your model?
What variable(s) are the feature(s) for your model?
Drop the EIN and NAME columns.
Determine the number of unique values for each column.
For columns that have more than 10 unique values, determine the number of data points for each unique value.
Use the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value, Other, and then check if the replacement was successful.
Use pd.get_dummies() to encode categorical variables.
Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.
Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

## Step 2: Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.
Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.
Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
Create the first hidden layer and choose an appropriate activation function.
If necessary, add a second hidden layer with an appropriate activation function.
Create an output layer with an appropriate activation function.
Check the structure of the model.
Compile and train the model.
Create a callback that saves the model's weights every five epochs.
Evaluate the model using the test data to determine the loss and accuracy.
Saved and exported results to an HDF5 file- AlphabetSoupCharity.h5.

## Step 3: Optimize the Model

Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.
Use any or all of the following methods to optimize your model:
Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
Dropping more or fewer columns.
Creating more bins for rare occurrences in columns.
Increasing or decreasing the number of values for each bin.
Add more neurons to a hidden layer.
Add more hidden layers.
Use different activation functions for the hidden layers.
Add or reduce the number of epochs to the training regimen.
Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.
Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.
Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.
Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.
Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.
Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

## Report: 


The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. 
Using the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

Step 1- I did data preprocessing, I Import and read the charity_data.csv. I dropped the non-beneficial ID columns, 'EIN' and 'NAME'. 
Determine the number of unique values in each column. Get value counts for the APPLICATION_TYPE column and CLASSIFICATION value counts.
Converted categorical data to numeric with `pd.get_dummies. Step 2- Split our preprocessed data into our features and target arrays.
Split the preprocessed data into a training and testing dataset. Create a StandardScaler instances. Fit the StandardScaler and Scale the data. 
Step 3- Compile train and evalute model. Deep neural net,
i.e., the number of input features and hidden nodes for each layer using nn = tf.keras.models.Sequential(). Compile the model. Train the model. Evaluate the model using the test data. 
I evaluated against performance metrics, I got accuracy = 0.72 and loss = 0.55. I did not get 90% accurancy, but accurancy depends on various factor.
Since I got specific performance metrics and target goals,
I am able to achieve target model performance. For more accurate result my suggestion would be we need to continue testing and evaluating this model.

The overall results of the deep learning model- Loss: 0.5555146336555481, Accuracy: 0.723498523235321. 
It is clear that there is room for improvement to reach the target accuracy of 90%. 
This suggests that while the model is able to learn from the data to some extent, it is not capturing the complexity or patterns sufficiently to reach the desired performance.  
The reported loss value indicates that there is still significant error in the predictions made by the model. To enhance the model's performance, 
there  are  alternative approach like Convolutional Neural Networks (CNNs). Different approach are for different types of data, like CNNs is for image or video recognition.
The model Fully Connected Neural Networks (FCNNs) we used is for structured/tabular data. So this is the best approach for deep learning model.
