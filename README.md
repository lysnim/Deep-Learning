
# Deep-Learning

[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.png?v=103)](https://opensource.org/licenses/mit-license.php)
[![Code Climate](https://codeclimate.com/github/MrRobb/Deep-Learning/badges/gpa.svg)](https://codeclimate.com/github/MrRobb/Deep-Learning)
[![Issue Count](https://codeclimate.com/github/MrRobb/Deep-Learning/badges/issue_count.svg)](https://codeclimate.com/github/MrRobb/Deep-Learning)
[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)


Here is all my progress through the arduous task of learning deep learning algorithms.

# Table of Contents
1. [Linear Regression Model](#body-brain-weight)
2. [Gradient Descent](#gradient-descent)

## Body-brain Weight
It explores the relationship between the weight of the body and the weight of the brain of the organism.

### Libraries
- [panda](http://pandas.pydata.org): a library to input, organize and manipulate data.
- [sklearn](http://scikit-learn.org): a library with simple and efficient tools for data mining and data analysis.
- [matplotlib](https://matplotlib.org): to plot data as we do in Matlab.

### Process
1. With panda we read the data and it returns a Dataframe type.
2. We select the x_values and the y_values indexed with the two first values.
3. With sklearn we create a linear regression model.
4. With sklearn we fit our linear model to the x and y values.
5. We scatter the points into the plot.
6. We add the line of our prediction.
7. We show our plot.

## Gradient Descent
It computes the optimized straight line that has the least amount of error.

## Neural Network
It creates a simple neural network of 1 neuron.

### Libraries
1. [Numpy](http://www.numpy.org)
2. Were also defining our own NeuralNetwork class.

### Process
1. We create our own NeuralNetwork class:
    1. We create 2 methods:
        1. **Sigmoid function** (S shaped curve, we pass the weighted sum of the inputs to normalize between 0 and 1)
        2. **Sigmoid derivative** function (To calculate the gradient)
    2. We make it think (We use it on training):
        1. Making the **sigmoid of the multiplication of the weights with the inputs**.
    3. We train our neuron:
        1. We make the NeuralNetwork think by passing the training set input values.
        2. Calculate the **error** (The difference between the real output and the predicted one)
        3. We **adjust the weights** by: multiplying (the inputs) by (the error by the derivate of the predicted output).
    4. We create an init method where we **seed the random values** and setting **random values to the weights**.
2. Create an instance of a neuron.
3. Print the random weights.
4. Set a training set.
5. Train the neuron with the training set.
6. Print the adjusted values of the neuron.
