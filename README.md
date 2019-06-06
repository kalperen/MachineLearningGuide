# MachineLearningGuide


## Table of Contents
- [Introduction](#introduction)
- [Linear Regression](#linear-regression)
- [Multivariate Linear Regression](#multivariate-linear-regression)

## Introduction
This repository contains my notes on the concepts of Machine Learning.

### What is Machine Learning:

(Tom Mitchell) Well-Posed Learning Problem: A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.

### Supervised Learning:
We are given a data set and know what the correct output should look like for each input.
Supervised Learning problems can be "regression" or "classification" problems.
***Regression problems*** consist of trying to predict results within a continuous output. i.e: Given a picture of a person, we have to predict their age on the basis of the given picture
***Classification problems*** consists of trying to predict results in a discrete output. i.e: Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.

### Unsupervised Learning:
Derive structure from data where we don't necessarily know the effect of the variables.

***Clustering***: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.

***Non-clustering***: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a cocktail party).

### Definitions:

***x***: “input” variables, also called input features

***y***: “output” or target variable that we are trying to predict.

A pair ***(x , y)***: is a training example, and the dataset that we’ll be using to learn—a list of m training examples (x(i),y(i));i=1,...,m—is called a training set.

***X*** and ***Y***: the space of input and output values.

***Hypothesis function***: In Supervised learning the goal is, given a training set, to learn a function h : X → Y so that h(x) is a “good” predictor for the corresponding value of y.

***Cost Function***: Used to measure the accuracy of the hypothesis function. It can take many forms but a commonly used one is the mean square error which takes an average difference of all results of the hypothesis with inputs from x's and the actual output y's as seen below:

![Alt text](/images/formulas/costFunction.png?raw=true "Cost Function")

***Gradient Descent****: A techinique to minimize loss by computing the gradients of loss with respect to the model's parameters, conditioned on training data.

![Alt text](/images/graphs/gradientDescent.png?raw=true "Gradient Descent")

With each step gradient descent adjusts parameters to find the best combination of weights and bias to minimize loss.

## Linear Regression:

Linear Regression is one of the simplest ML models. We have a hypothesis set with only two parameters that we trying to tune in order to minimize the cost function. Informally this means that given an input x and an output y we want our model's prediction to be as close to y as possible (for multiple pairs x and y). To achieve this we use gradient descent algorithm defined below.

![Alt text](/images/models/linearRegression/linearRegression.png?raw=true "Linear Regression")
![Alt text](/images/models/linearRegression/gradientDescent.png?raw=true "Linear Regression")

Usually when learning about Linear Regression we consider the example of trying to predict the price of a home given some information about it.

## Multivariate Linear Regression:

Multivariate Linear Regression is the same as regular Linear Regression but with more parameters. Considering the case of trying to predict the price of a house, while Linear Regression may only have a single parameter (size of home), multivariate Linear Regression can take into account multiple parameters (size of home, number of rooms, location). To achieve this we just have to slightly modify the Hypothesis and Gradient Descent to take into account multiple parameters.

![Alt text](/images/models/linearRegression/MLinRegHyp.png?raw=true "Multivariate Linear Regression")
![Alt text](/images/models/linearRegression/MLinRegGrad.png?raw=true "Multivariate Linear Regression")

Having multiple parameters can create problems for gradient descent and lead to slow convergence. One common problem is having parameters that have widely different ranges. For example size of house can be between 0 and 1000 while number of rooms is between 0 and 5 in a certain data set. To solve this problem and allow gradient descent to converge faster we use two techniques called feature scaling and mean normalization.

***Feature Scaling***: dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1.

***Mean Normalization***: subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero.

To implement both these techniques we modify the input variables as follows:

![Alt text](/images/formulas/FScaleMNorm.png?raw=true "Feature Scaling and Mean Normalization")

Even with these adjustments its still possible for Gradient Descent not to converge. This may be due to a bad learning rate. If the learning rate is too small convergence may take too long, if the learning rate is too large the cost might not decrease on every iteration and gradient descent thus might not converge.

To debug gradient descent make a plot with number of iterations on the x-axis. Now plot the cost function, J(θ) over the number of iterations of gradient descent. If J(θ) ever increases, then you probably need to decrease the learning rate.

***Feature Crossing***:

Its possible to combine multiple features into one and create a feature cross, this technique is useful in a multitude of scenarios. For example when predicting the price of a house we may want to create a feature cross of latitude and number of bedrooms, this will allow us to learn that maybe having three bedrooms in Sand Francisco is not the same as having three bedrooms in Sacramento.

In practice Feature Crossing is seldom used for continuous features. A more common use case is feature crosses of one-hot vectors. For example, suppose we have two features: country and language. A one-hot encoding of each generates vectors with binary features that can be interpreted as country=USA, country=France or language=English, language=Spanish. Then, if you do a feature cross of these one-hot encodings, you get binary features that can be interpreted as logical conjunctions, such as: country:usa AND language:spanish.

***Normal Equation***:

Gradient descent gives one way of minimizing J. Let’s discuss a second way of doing so, this time performing the minimization explicitly and without resorting to an iterative algorithm. In the "Normal Equation" method, we will minimize J by explicitly taking its derivatives with respect to the θj ’s, and setting them to zero. This allows us to find the optimum theta without iteration.

![Alt text](/images/formulas/NormalEquation.png?raw=true "Normal Equation")

It's possible for the X'X in the formula to be noninvertible due to redundant or too many features. To solve this problem we can delete some features or use regularization.

There is no need to do feature scaling with the normal equation.

![Alt text](/images/graphs/GDvsNE.png?raw=true "Gradient Descent vs Normal Equation")
