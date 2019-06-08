# MachineLearningGuide


## Table of Contents
- [Introduction](#introduction)
- [Linear Regression](#linear-regression)
- [Multivariate Linear Regression](#multivariate-linear-regression)

## Introduction
This repository contains my notes on the concepts of Machine Learning.

***Credits to***:

https://www.coursera.org/learn/machine-learning/
https://developers.google.com/machine-learning/crash-course/

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

## Logistic Regression

***Classification***:

Predicting for classification is similar to doing regression except that the values we now want take on only a small number of discrete values. The simplest form of classification is binary classification in which y can take only two values, 0 and 1. An example of binary classification is to try to predict whether an email is spam or not.

***Hypothesis***:

Since we are now predicting for a class it doesn't make sense for our hypothesis function to take values bigger than 1. We will therefore fix the form of our hypotheses by plugging θ'x into the logistic function and using the sigmoid function.

![Alt text](/images/models/logisticRegression/hypothesis.png?raw=true "Hypothesis")

![Alt text](/images/graphs/sigmoid.png?raw=true "Sigmoid")
The sigmoid function g(z) maps any real number to the (0, 1) interval.

The hypothesis function h(x) will give us the probability that our output is 1.

![Alt text](/images/models/logisticRegression/hypoProba.png?raw=true "Hypothesis")

***Decision Boundary***:

In order to get our discrete 0 or 1 classification, we can translate the output of the hypothesis function using a decision Boundary. The decision boundary is the line that separates the area where y = 0 and where y = 1. It is created by our hypothesis function.

***Cost Function***:

For logistic regression we use log loss as the cost function defined as follows:

![Alt text](/images/models/logisticRegression/costFunction.png?raw=true "Cost Function")

![Alt text](/images/models/logisticRegression/costFunction2.png?raw=true "Cost Function Simplified")

***Multiclass Classification (One vs all)***:

Instead of having y = {0, 1} we now have y = {0,1,...n}. Because we now have multiple classes we divide our problem into n + 1 binary classification problems. In each case we predict the probability that 'y' is a member of one of our classes. Informally, we are choosing one class and then lumping all the others into a single second class. After doing this repeatedly by applying binary logistic regression to each case we use the hypothesis that returned the highest value as our prediction.

![Alt text](/images/models/logisticRegression/multiClass.png?raw=true "Multiclass classification")

## Regularization and solving the problem of overfitting

Underfitting, or high bias, is when the form of our hypothesis function h maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features. At the other extreme, overfitting, or high variance, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

There are two main options to address the issue of overfitting:

1) Reduce the number of features:

- Manually select which features to keep.
- Use a model selection algorithm (studied later in the course).

2) Regularization
- Keep all the features, but reduce the magnitude of parameters θj.
- Regularization works well when we have a lot of slightly useful features.

***L2 regularization and Lambda***:
Our training optimization algorithm is now a function of two terms: the loss term, which measures how well the model fits the data, and the regularization term, which measures model complexity.

We can quantify complexity using the L2 regularization formula, which defines the regularization term as the sum of the squares of all the feature weights:

![Alt text](/images/formulas/l2reg.png?raw=true "L2 Regularization")

In this formula, weights close to zero have little effect on model complexity, while outlier weights can have a huge impact.

If we have overfitting from our hypothesis function, we can reduce the weight that some of the terms in our function carry by increasing their cost. To do this model developers tune the overall impact of the regularization term by multiplying its value by a scalar known as lambda (also called the regularization rate). That is, model developers aim to do the following: minimize(Loss(Data|Model) + λcomplexity(Model)).

Performing L2 regularization has the following effect on a model:

- Encourages weight values toward 0 (but not exactly 0)
- Encourages the mean of the weights toward 0, with a normal (bell-shaped or Gaussian) distribution.

Increasing the lambda value strengthens the regularization effect.

When choosing a lambda value, the goal is to strike the right balance between simplicity and training-data fit:

- If your lambda value is too high, your model will be simple, but you run the risk of underfitting your data. Your model won't learn enough about the training data to make useful predictions.
- If your lambda value is too low, your model will be more complex, and you run the risk of overfitting your data. Your model will learn too much about the particularities of the training data, and won't be able to generalize to new data.

The ideal value of lambda produces a model that generalizes well to new, previously unseen data. Unfortunately, that ideal value of lambda is data-dependent, so you'll need to do some tuning.

***Regularized Linear Regression***:
To regularize linear regression we modify gradient descent to separate the bias and add the regularization term to the rest of the parameters as follows:

![Alt text](/images/models/linearRegression/regularized.png?raw=true "Regularized Linear Regression")

For the normal equation the transformation is as follows:
![Alt text](/images/models/linearRegression/regularizedNorm.png?raw=true "Regularized Normal Equation")

***Regularized Logistic Regression***:

To regularize logistic regression we modify the cost function as follows:

![Alt text](/images/models/logisticRegression/regularized.png?raw=true "Regularized Logistic Regression")
