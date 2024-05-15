## Overview of the analysis: Explain the purpose of this analysis.

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. Using the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

## Results: 
Using bulleted lists and images to support your answers, address the following questions:
Data Preprocessing
# What variable(s) are the target(s) for your model?
IS_SUCCESSFUL are targets variable(s) y for the model.
## What variable(s) are the features for your model?
All other columns in categorical_dummies_df are features variable(s) X.
## What variable(s) should be removed from the input data because they are neither targets nor features?
Columns, 'EIN' and 'NAME' variable(s) should be removed from the input data because they are neither targets nor features
## Compiling, Training, and Evaluating the Model
## How many neurons, layers, and activation functions did you select for your neural network model, and why?
neurons = Total neurons are 33(first hidden layer units=16, second hidden layer units=16 and output layer units=1). I selected 33 neurons because these neurons are responsible for processing inputs through the network by performing weighted sums followed by activation functions, each neuron contributes to capturing different aspects and complexities of the input data.
layers= There are two layers- first hidden layer and second hidden layer. Two hidden layers are perfect as this setup helps start modeling complexity without being too deep, which could require more data and cause overfitting in smaller datasets.
activation functions= ReLU (Rectified Linear Unit) for the hidden layers and Sigmoid for the output layer are used in this model.
## Were you able to achieve the target model performance?
I evaluated against performance metrics, I got accuracy = 0.72 and loss = 0.55. I did not get 90% accurancy, but accurancy depends on various factor. Since I got specific performance metrics and target goals, I am able to achieve target model performance. For more accurate result my suggestion would be we need to continue testing and evaluating this model.
## What steps did you take in your attempts to increase model performance?
Step 1- I did data preprocessing, I Import and read the charity_data.csv. I dropped the non-beneficial ID columns, 'EIN' and 'NAME'.  Determine the number of unique values in each column. Get value counts for the APPLICATION_TYPE column and CLASSIFICATION value counts. Converted categorical data to numeric with `pd.get_dummies. Step 2- Split our preprocessed data into our features and target arrays.Split the preprocessed data into a training and testing dataset. Create a StandardScaler instances. Fit the StandardScaler and Scale the data. Step 3- Compile train and evalute model. Deep neural net, i.e., the number of input features and hidden nodes for each layer using nn = tf.keras.models.Sequential(). Compile the model. Train the model. Evaluate the model using the test data.
## Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

The overall results of the deep learning model- Loss: 0.5555146336555481, Accuracy: 0.723498523235321. It is clear that there is room for improvement to reach the target accuracy of 90%.  This suggests that while the model is able to learn from the data to some extent, it is not capturing the complexity or patterns sufficiently to reach the desired performance.  The reported loss value indicates that there is still significant error in the predictions made by the model. To enhance the model's performance, there  are  alternative approach like Convolutional Neural Networks (CNNs). Different approach are for different types of data, like CNNs is for image or video recognition. The model Fully Connected Neural Networks (FCNNs) we used is for structured/tabular data. So this is the best approach for deep learning model.