# Deep Learning in Python

# Introduction

## Activation Functions

* An activation function allows the nodes to capture non-linearities and capture patterns.
* It is applied to a value coming into the node. It converts the node's input into some output.
* Example: Rectified Linear Activation activation function (ReLU). This function takes a single number as an input, returning 0 if the input is negative, and the input if the input is positive.

relu(3) = 3
relu(-3) = 0
<image>

Identity Function: Returns the same output as the input.

## Deeper Networks

* Deep Networks consist of more than 1 hidden layers. The same forward propagation process is used across all the hidden layers.
* Deep networks internally build representations of patterns in the data.
* They partially replace the need for feature engineering since subsequent layers build better representations of the raw data.
* For example, the first hidden layer may identify a line, the second may identify a diagonal, the third may identify a square, and subsequent layers may come up with more complex geometrical shapes based on the data.
* An advantage is that the modeler need not specify these interactions.

<image>

# Optimizing Neural Networks

* The output of a neural network depends on the weights assigned at each input at each layer.
* A change in weights may change the outputs to make more accurate predictions.
* Making accurate predictions gets more difficult with more data points.
* At any set of weights there may be many values of the error corresponding to the data points the predictions are made for.

## Loss Function

* A function that aggregates the errors in predictions from many data points into a single number. It gives a measure of the model's predictive performance.
* Low value of the loss function indicates a better model performance.
* Example: Mean Squared Error - squares all the errors and calculates the mean.

**Goal of Optimization:** Find weights such that we get the lowest value of the loss function, i.e. the error is minimised.

## Gradient Descent

* A popular optimization algorithm is the Gradient Descent.
* Following are the steps of this algorithm:
1. Start at a random point.
2. Calculate the slope at this point.
3. Move in a direction opposite the slope.
4. Repeat until the slope is somewhat flat.

* This algorithm eventually leads to the minimum value. Hence, it is used to minimise the value of the loss function.
<image>

### Learning Rate

* If the slope is positive: going opposite to the slope means, moving to lower numbers, i.e. Subtracting the slope from the current value.
* In this, the problem is that, if the step is too big, we may move far astray (overshooting minima).
* Hence, we multiply the slope by a small value called learning rate.
* Update the current value by subtracting (learning rate x slope) from the current value.

### Slope Calculation

Consider an example:

        2
    3------->6

Here the input 3, feeds into the output with a weight of 2.
Let, Actual Target=10
Here, the prediction is 6.
Error = 6-10 = -4

To calculate the slope for a weight, we multiply:

* **Slope of the loss function w.r.t the value of the node we feed into.**

In this case, this is the output node 6.
We can observe that, 
2(Predicted-Actual) = 2(Error)  => 2*(-4)

* **The value of the node that feeds into our weight.**
Here, this node is the node 3.

* **Slope of the activation function w.r.t the value we feed into.**
Here, there no activation function and hence can be ignored.

Hence, the slope is given as:

    Slope=2 * (-4) * 3= -24

Now, since we have the slope, we can calculate the new weight.
Consider the learning rate=0.01

    New Weight = Old Value - (learning_rate * Slope)

New weight = 2 - (0.01 * -24) = 2.24

This process is repeated for each weight.
In the end, all the weights are updated *simultaneously*.

## Backpropagation

* This technique propagates the error from the output through the hidden layers to the input layer.
* It allows gradient descent to update all weights in a neural network.

**Process:**
* Estimate the slope of the loss function w.r.t each weight.
* Use forward propagation to calculate the predictions and errors before doing back propagation.
* Go back one layer at a time.
* Calculate the gradient as the product of the three terms described in the above section.
* Need to keep track of the slopes w.r.t the values of the nodes.
* Slope of node values = sum of the slopes of the weights coming out from the node.

## Stochastic Gradient Descent

* Calculates slopes only on a subset of data.
* Use a different batch of data to calculate the next update.
* Start over from the beginning once all the data is used.
* Each time through the training data is called an **Epoch**.