# Machine Learning Guide

***THIS IS A WORK IN PROGRESS***

This repository contains my notes on the concepts of Machine Learning. I use it to keep track of my learnings but it can also be applied as a general guide on how to get started with Machine Learning. Be careful as the depth of notes is not consistent throughout the document, I get into more detail wherever I deem necessary and only present the general ideas in other parts. Feel free to contribute with pull requests!

***Credits to***:

https://www.coursera.org/learn/machine-learning/

https://developers.google.com/machine-learning/crash-course/

https://github.com/ageron/handson-ml (notebooks are mostly based on exercises completed from the Hands-On Machine Learning With Scikit-Learn & TensorFlow book)

## Table of Contents
- [Learning Machine Learning](#learning-machine-learning)
- [Linear Regression](#linear-regression)
- [Multivariate Linear Regression](#multivariate-linear-regression)
- [Logistic Regression](#logistic-regression)
- [Regularization and Overfitting](#regularization-and-overfitting)
- [Neural Networks](#neural-networks)

### Learning Machine Learning

There are tons of resources out there on how to learn ML. This is the path I took:

***Math***

For a general review you can follow this course:

https://www.youtube.com/playlist?list=PL7y-1rk2cCsAqRtWoZ95z-GMcecVG5mzA

Linear Algebra:

For a brief conceptual introduction follow through this series:

https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab

Then follow through this course for more in depth teachings:

https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/video-lectures/

Calculus:

For a brief conceptual introduction follow through this series:

https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr

Then follow through these courses for more in depth teachings:

https://ocw.mit.edu/courses/mathematics/18-01-single-variable-calculus-fall-2006/video-lectures/

https://ocw.mit.edu/courses/mathematics/18-02-multivariable-calculus-fall-2007/video-lectures/

Probability and Statistics:

http://web.stanford.edu/class/archive/cs/cs109/cs109.1166//handouts/overview.html

If you're more into text books then I recommend this:

https://mml-book.github.io/

***ML***

For a lightweight introduction to ML follow this google tutorial:

https://developers.google.com/machine-learning/crash-course/

For a more formal introduction you can take the most famous ML course out there:

https://www.coursera.org/learn/machine-learning/

If you want to continue studying further reading these is a good first step:

https://www.deeplearningbook.org/

http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf


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

## Regularization and Overfitting

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

## Neural Networks

***Simplified Explanation***

A Neural Network is a machine learning model commonly used to solve nonlinear problems. A Neural Network is composed of:
- A set of nodes, analogous to neurons, organized in layers.
- A set of weights representing the connections between each neural network layer and the layer beneath it. The layer beneath may be another neural network layer, or some other kind of layer.
- A set of biases, one for each node.
- An activation function that transforms the output of each node in a layer. Different layers may have different activation functions.

![Alt text](/images/models/neuralNetworks/graph.png?raw=true "Simple Neural Network")

Each blue circle represents an input feature, and the green circle represents the weighted sum of the inputs.
Each yellow node in the hidden layer is a weighted sum of the blue input node values. The output is a weighted sum of the yellow nodes.
The nodes in pink represent the hidden layer used to introduce nonlinearity by piping each hidden layer node thorugh a nonlinear function.
With an activation function, adding layers has more impact. Stacking nonlinearities on nonlinearities lets us model very complicated relationships between the inputs and the predicted outputs. In brief, each layer is effectively learning a more complex, higher-level function over the raw inputs.

Common activation functions are Sigmoid and Rectified linear unit.
Sigmoid: F(x) = 1/(1 + e^-x)
ReLu: F(x) = max(0, x)

***Formal Definition***
At a very simple level, neurons are basically computational units that take inputs (dendrites) as electrical inputs (called "spikes") that are channeled to outputs (axons). In our model, our dendrites are like the input features x1⋯xn, and the output is the result of our hypothesis function. In this model our x0 input node is sometimes called the "bias unit." It is always equal to 1. In neural networks, we use the same logistic function as in classification, 1/(1+e^(-θ'x')), yet we sometimes call it a sigmoid (logistic) activation function. In this situation, our "theta" parameters are sometimes called "weights".

Our input nodes (layer 1), also known as the "input layer", go into another node (layer 2), which finally outputs the hypothesis function, known as the "output layer".

We can have intermediate layers of nodes between the input and output layers called the "hidden layers."

![Alt text](/images/models/neuralNetworks/definitions.png?raw=true "Definitions")

![Alt text](/images/models/neuralNetworks/representation.png?raw=true "Representation")

We compute our activation nodes by using a 3×4 matrix of parameters. We apply each row of the parameters to our inputs to obtain the value for one activation node. Our hypothesis output is the logistic function applied to the sum of the values of our activation nodes, which have been multiplied by yet another parameter matrix Θ(2) containing the weights for our second layer of nodes.

![Alt text](/images/models/neuralNetworks/activation.png?raw=true "Activaton")

Each layer gets its own matrix of weights, Θ(j).

The dimensions of these matrices of weights is determined as follows:

If network has sj units in layer j and sj+1 units in layer j+1, then Θ(j) will be of dimension sj+1×(sj+1).

To simplify the above definition of the activation funtions we define a new variable z that encompasses the parameters inside our g function.

We can now define our hypothesis function as follows:

![Alt text](/images/models/neuralNetworks/hypothesis.png?raw=true "Hypothesis")

To classify data into multiple classes, we let our hypothesis function return a vector of values.


## Simple Q&A

***1. Define Machine learning***

ML is building systems that can learn from data by getting better at some task given some performance measure.

***2. In what type of problems does ML shine?***

ML is good for complex problems for which we have no algorithmic solution, to replace long lists of hand-tuned rules, to build systems that adapt to fluctuating environments, and to help humans learn from data.

***3. What are the most common supervised learning tasks?***

Regression and Classification.

***4. What are the most common unsupervised learning tasks?***

Clustering, Visualization, Dimensionality Reduction, and Association Rule Learning.

***5. What is an online learning system?***

An online learning systems learns incrementally as opposed to a batch-learning system. It can adapt rapidly to both changing data and autonomous systems and can be trained on very large quantities of data.

***6. What is out-of-core learning?***

Out-of-core learning systems are used to handle vast quantities of data that cannot fit in a computer's main memory. They work by chopping the data into mini-batches and using online learning techniques to learn from these mini-batches.

***7. What type of learning algo relies on a similarity measure to make predictions?***

Instance-based learning systems learn the data by heart then when given a new instance use a similarity measure to find the most similar learned instances to make predictions.

***8. What is the difference between a model parameter and a learning algorithm's hyperparameter?***\

A model has one or more model parameters that determine what it will predict given a new instance (slope of a linear model). A learning algorithm tries to find optimal values for these parameters such that the model generalizes well to new instances. A hyperparameter is a parameter of the learning algorithm itself, not of the model (the amount of regularization to apply)./

***9.What do model-based learning algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?***

Model-based learning algorithms search for an optimal value for the model parameters such that the model will generalize well to new instances. We usually train such systems by minimizing a cost function that measures how bad the system is at making predictions on the training data, plus a penalty for model complexity if the model is regularized. To make predictions we feed the new instance's features into the model's prediction function, using the parameter values found by the learning algorithm.

***10. What are the main challenges in ML?***

Lack of data, poor data quality, non-representative data, uninformative features, excessively simple models that underfit the training data, and excessively complex models that overfit the data.

***11. If you model performs great on training data but generalizes poorly to new instances, what is happening? What are possible solutions?***

The model is likely overfitting the training data. Possible solutions are getting more data, simplifying the model (selecting simpler algorithm, reducing the number of parameters or features used, or regularizing the model), or reducing the noise in the data.

***12. What is a test set and when is it used?***

Test sets are used to estimate the generalization error that a model will make on new instances, before the model is launched in production.

***13. What is a validation set and when is it used?***

A validation set is used to compare models, it makes it possible to select the best model and tune the hyperparameters.

***14. What happens if you tune hyperparameters using the test set?***

You risk overfitting the test set, and the generalization error you measure will be optimistic (launch a model that performs worse than you expect it to).

***15. What is cross-validation adn why would you prefer it to a validation set?***

Cross-validation is a technique that makes it possible to compare models (for model selection and hyperparameter tuning) without the need for a separate validation set. This saves training data.

***16. What Linear Regression training algorithm can you use if you have a training set with millions of features?***

If you have a training set with millions of features you can use Stochastic Gradient Descent or Mini-batch Gradient Descent, and perhaps Batch Gradient Descent if the training set fits in memory. But you cannot use the Normal Equation because the computational complexity grows quickly (more than quadratically) with the number of features.

***17. Suppose the features in you training set have  very different scales. What algos might suffer from this, and how? What can you do about it?***

If the features in your training set have very different scales, the cost function will have the shape of an elongated bowl, so the Gradient Descent algorithms will take a long time to converge. To solve this you should scale the data before training the model. Note that the Normal Equation will work just fine without scaling.

***18. an Gradient Descent get stuck in a local minimum when training a Logistic Regression model?***

Gradient Descent cannot get stuck in a local minimum when training a Logistic Regression model because the cost function is convex

***19. Do all Gradient Descent algorithms lead to the same model provided you let them run long enough?***

If the optimization problem is convex (such as Linear Regression or Logistic Regression), and assuming the learning rate is not too high, then all Gradient Descent algorithms will approach the global optimum and end up producing fairly similar models. However, unless you gradually reduce the learning rate, Stochastic GD and Mini-batch GD will never truly converge; instead, they will keep jumping back and forth around the global optimum. This means that even if you let them run for a very long time, these Gradient Descent algorithms will produce slightly different models.

***20. Suppose you use Batch Gradient Descent and you plot the validation error at every epoch. If you notice that the validation error consistently goes up, what is likely going on? How can you fix this?***

If the validation error consistently goes up after every epoch, then one possibility is that the learning rate is too high and the algorithm is diverging. If the training error also goes up, then this is clearly the problem and you should reduce the learning rate. However, if the training error is not going up, then your model is overfitting the training set and you should stop training.

***21. Is it a good idea to stop Mini-batch Gradient Descent immediately when the validation error goes up?***

Due to their random nature, neither Stochastic Gradient Descent nor Mini-batch Gradient Descent is guaranteed to make progress at every single training itera‐ tion. So if you immediately stop training when the validation error goes up, you may stop much too early, before the optimum is reached. A better option is to save the model at regular intervals, and when it has not improved for a long time (meaning it will probably never beat the record), you can revert to the best saved model.

***22. Which Gradient Descent algo will reach the vicinity of the optimal solution the fastest? Which will actually converge? How can you make others converge as well?***

Stochastic Gradient Descent has the fastest training iteration since it considers only one training instance at a time, so it is generally the first to reach the vicinity of the global optimum (or Mini-batch GD with a very small mini-batch size). However, only Batch Gradient Descent will actually converge, given enough training time. As mentioned, Stochastic GD and Mini-batch GD will bounce around the optimum, unless you gradually reduce the learning rate.

***23. Suppose you are using Polinomial Regression. You plot the learning curves and you notice that there is a large gap between the training error and the validation error. What is happening? What are three ways to solve this?***

If the validation error is much higher than the training error, this is likely because your model is overfitting the training set. One way to try to fix this is to reduce the polynomial degree: a model with fewer degrees of freedom is less likely to overfit. Another thing you can try is to regularize the model—for example, by adding an ℓ2  penalty (Ridge) or an ℓ1  penalty (Lasso) to the cost function. This will also reduce the degrees of freedom of the model. Lastly, you can try to increase the size of the training set

***24. Suppose you are using ggRidge Regression and you notice that the training error and the validation error are almost equal and fairly higgh. Would you say that the model suffers from high bias or high variance? Should you increase the regularization hyperparameter alpha or reduce it?***

If both the training error and the validation error are almost equal and fairly high, the model is likely underfitting the training set, which means it has a high bias. You should try reducing the regularization hyperparameter α.

***25. Why would you want to use: Ridge Regression instead of plain Linear Reggression? Lasso instead of Ridge Regression? Elastic Net instead of Lasso?***

Let’s see:

- A model with some regularization typically performs better than a model without any regularization, so you should generally prefer Ridge Regression over plain Linear Regression.

- Lasso Regression uses an ℓ1  penalty, which tends to push the weights down to exactly zero. This leads to sparse models, where all weights are zero except for the most important weights. This is a way to perform feature selection auto‐ matically, which is good if you suspect that only a few features actually matter. When you are not sure, you should prefer Ridge Regression.

- Elastic Net is generally preferred over Lasso since Lasso may behave erratically in some cases (when several features are strongly correlated or when there are more features than training instances). However, it does add an extra hyper‐ parameter to tune. If you just want Lasso without the erratic behavior, you can just use Elastic Net with an l1_ratio close to 1.

***26. Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime. Should you implement two Logistic Regression classifiers or one Softmax Regression classifier?***

If you want to classify pictures as outdoor/indoor and daytime/nighttime, since these are not exclusive classes (i.e., all four combinations are possible) you should train two Logistic Regression classifiers.


***27. What is the fundamental idea behind Support Vector Machines?***

The fundamental idea behind Support Vector Machines is to fit the widest possible “street” between the classes. In other words, the goal is to have the largest possible margin between the decision boundary that separates the two classes and the training instances. When performing soft margin classification, the SVM searches for a compromise between perfectly separating the two classes and having the widest possible street (i.e., a few instances may end up on the street). Another key idea is to use kernels when training on nonlinear datasets.

***28. What is a support vector?***

After training an SVM, a support vector is any instance located on the “street” (see the previous answer), including its border. The decision boundary is entirely determined by the support vectors. Any instance that is not a support vector (i.e., off the street) has no influence whatsoever; you could remove them, add more instances, or move them around, and as long as they stay off the street they won’t affect the decision boundary. Computing the predictions only involves the support vectors, not the whole training set.

***29. Why is it important to scale the inputs when using SVMs?***

SVMs try to fit the largest possible “street” between the classes (see the first answer), so if the training set is not scaled, the SVM will tend to neglect small features.

***30. Can an SVM classifier output a confidence score when it classifies an instance? What about a probability?***

An SVM classifier can output the distance between the test instance and the decision boundary, and you can use this as a confidence score. However, this score cannot be directly converted into an estimation of the class probability. If you set probability=True when creating an SVM in Scikit-Learn, then after training it will calibrate the probabilities using Logistic Regression on the SVM’s scores (trained by an additional five-fold cross-validation on the training data). This will add the predict_proba() and predict_log_proba() methods to the SVM.

***31. Should you use the primal or the dual form of the SVM problem to train a model on a training set with millions of instances and hundreds of features?***

This question applies only to linear SVMs since kernelized can only use the dual form. The computational complexity of the primal form of the SVM problem is proportional to the number of training instances m, while the computational complexity of the dual form is proportional to a number between m2 and m3. So if there are millions of instances, you should definitely use the primal form, because the dual form will be much too slow.

***32. Say you trained an SVM classifier with an RBF kernel. It seems to underfit the training set: should you increase or decrease γ (gamma)? What about C?***

If an SVM classifier trained with an RBF kernel underfits the training set, there might be too much regularization. To decrease it, you need to increase gamma or C (or both).

***33. How should you set the QP parameters (H, f, A, and b) to solve the soft margin linear SVM classifier problem using an off-the-shelf QP solver?***

Let’s call the QP parameters for the hard-margin problem H′, f′, A′ and b′. The QP parameters for the soft-margin problem have m additional parameters (np = n + 1 + m) and m additional constraints (nc = 2m). They can be defined like so:
- H is equal to H′, plus m columns of 0s on the right and m rows of 0s at the bottom
- f is equal to f′ with m additional elements, all equal to the value of the hyperparameter
C.
-  b is equal to b′ with m additional elements, all equal to 0.
- A is equal to A′, with an extra m × m identity matrix Im appended to the right,

***34. What is the approximate depth of a Decision Tree trained (without restrictions) on a training set with 1 million instances?***

The depth of a well-balanced binary tree containing m leaves is equal to log2(m)3, rounded up. A binary Decision Tree (one that makes only binary decisions, as is the case of all trees in Scikit-Learn) will end up more or less well balanced at the end of training, with one leaf per training instance if it is trained without restrictions. Thus, if the training set contains one million instances, the Decision Tree will have a depth of log2(106) ≈ 20 (actually a bit more since the tree will generally not be perfectly well balanced).

***35. Is a node’s Gini impurity generally lower or greater than its parent’s? Is it generally lower/greater, or always lower/greater?***

A node’s Gini impurity is generally lower than its parent’s. This is ensured by the CART training algorithm’s cost function, which splits each node in a way that minimizes the weighted sum of its children’s Gini impurities. However, if one child is smaller than the other, it is possible for it to have a higher Gini impurity than its parent, as long as this increase is more than compensated for by a decrease of the other child’s impurity. For example, consider a node containing four instances of class A and 1 of class B. Its Gini impurity is 1 − 1/5^2 − 4/5^2 = 0.32. Now suppose the dataset is one-dimensional and the instances are lined up in the following order: A, B, A, A, A. You can verify that the algorithm will split this node after the second instance, producing one child node with instances A, B, and the other child node with instances A, A, A. The first child node’s Gini impurity is 1 − 1/2^2 − 1/2^2 = 0.5, which is higher than its parent. This is compensated for by the fact that the other node is pure, so the overall weighted Gini impurity is 25 × 0.5 + 35 × 0 = 0.2 , which is lower than the parent’s Gini impurity.

***36. If a Decision Tree is overfitting the training set, is it a good idea to try decreasing max_depth?***

If a Decision Tree is overfitting the training set, it may be a good idea to decrease max_depth, since this will constrain the model, regularizing it

***37. If a Decision Tree is underfitting the training set, is it a good idea to try scaling the input features?***

Decision Trees don’t care whether or not the training data is scaled or centered; that’s one of the nice things about them. So if a Decision Tree underfits the training set, scaling the input features will just be a waste of time.

***38. If it takes one hour to train a Decision Tree on a training set containing 1 million instances, roughly how much time will it take to train another Decision Tree on a training set containing 10 million instances?***

The computational complexity of training a Decision Tree is O(n × m log(m)). So if you multiply the training set size by 10, the training time will be multiplied by K = (n × 10m × log(10m)) / (n × m × log(m)) = 10 × log(10m) / log(m). If m = 106, then K ≈ 11.7, so you can expect the training time to be roughly 11.7 hours

***39. If your training set contains 100,000 instances, will setting presort=True speed up training?***

Presorting the training set speeds up training only if the dataset is smaller than a few thousand instances. If it contains 100,000 instances, setting presort=True will considerably slow down training.

***40. If you have trained five different models on the exact same training data, and they all achieve 95% precision, is there any chance that***

If you have trained five different models and they all achieve 95% precision, you can try combining them into a voting ensemble, which will often give you even better results. It works better if the models are very different (e.g., an SVM classifier, a Decision Tree classifier, a Logistic Regression classifier, and so on). It is even better if they are trained on different training instances (that’s the whole point of bagging and pasting ensembles), but if not it will still work as long as the models are very different.

***41. What is the difference between hard and soft voting classifiers?***
A hard voting classifier just counts the votes of each classifier in the ensemble and picks the class that gets the most votes. A soft voting classifier computes the average estimated class probability for each class and picks the class with the highest probability. This gives high-confidence votes more weight and often performs better, but it works only if every classifier is able to estimate class probabilities (e.g., for the SVM classifiers in Scikit-Learn you must set probability=True).

 ***42. Is it possible to speed up training of a bagging ensemble by distributing it across multiple servers? What about pasting ensembles, boosting ensembles, random forests, or stacking ensembles?***
It is quite possible to speed up training of a bagging ensemble by distributing it across multiple servers, since each predictor in the ensemble is independent of the others. The same goes for pasting ensembles and Random Forests, for the same reason. However, each predictor in a boosting ensemble is built based on the previous predictor, so training is necessarily sequential, and you will not gain anything by distributing training across multiple servers. Regarding stacking ensembles, all the predictors in a given layer are independent of each other, so they can be trained in parallel on multiple servers. However, the predictors in one layer can only be trained after the predictors in the previous layer have all been trained.

 ***43. What is the benefit of out-of-bag evaluation?***

 With out-of-bag evaluation, each predictor in a bagging ensemble is evaluated using instances that it was not trained on (they were held out). This makes it possible to have a fairly unbiased evaluation of the ensemble without the need for an additional validation set. Thus, you have more instances available for training, and your ensemble can perform slightly better.

 ***44. What makes Extra-Trees more random than regular Random Forests? How can this extra randomness help? Are Extra-Trees slower or faster than regular Random Forests?***

 When you are growing a tree in a Random Forest, only a random subset of the features is considered for splitting at each node. This is true as well for Extra- Trees, but they go one step further: rather than searching for the best possible thresholds, like regular Decision Trees do, they use random thresholds for each feature. This extra randomness acts like a form of regularization: if a Random Forest overfits the training data, Extra-Trees might perform better. Moreover, since Extra-Trees don’t search for the best possible thresholds, they are much faster to train than Random Forests. However, they are neither faster nor slower than Random Forests when making predictions.

***45. If your AdaBoost ensemble underfits the training data, what hyperparameters should you tweak and how?***

 If your AdaBoost ensemble underfits the training data, you can try increasing the number of estimators or reducing the regularization hyperparameters of the base estimator. You may also try slightly increasing the learning rate.

***46. If your Gradient Boosting ensemble overfits the training set, should you increase or decrease the learning rate***

If your Gradient Boosting ensemble overfits the training set, you should try decreasing the learning rate. You could also use early stopping to find the right number of predictors (you probably have too many).

***47. What are the main motivations for reducing a dataset’s dimensionality? What are
the main drawbacks?***

Motivations and drawbacks:
• The main motivations for dimensionality reduction are:
— To speed up a subsequent training algorithm (in some cases it may even remove noise and redundant features, making the training algorithm perform better).
— To visualize the data and gain insights on the most important features.
— Simply to save space (compression).
• The main drawbacks are:
— Some information is lost, possibly degrading the performance of subsequent training algorithms.
— It can be computationally intensive.
— It adds some complexity to your Machine Learning pipelines. —Transformed features are often hard to interpret.

***48. What is the curse of dimensionality?***

The curse of dimensionality refers to the fact that many problems that do not exist in low-dimensional space arise in high-dimensional space. In Machine Learning, one common manifestation is the fact that randomly sampled highdimensional vectors are generally very sparse, increasing the risk of overfitting and making it very difficult to identify patterns in the data without having plenty of training data.

***49. Once a dataset’s dimensionality has been reduced, is it possible to reverse the operation? If so, how? If not, why?***

Once a dataset’s dimensionality has been reduced using one of the algorithms we discussed, it is almost always impossible to perfectly reverse the operation, because some information gets lost during dimensionality reduction. Moreover, while some algorithms (such as PCA) have a simple reverse transformation procedure that can reconstruct a dataset relatively similar to the original, other algorithms (such as T-SNE) do not.

***50. Can PCA be used to reduce the dimensionality of a highly nonlinear dataset?***

PCA can be used to significantly reduce the dimensionality of most datasets, even if they are highly nonlinear, because it can at least get rid of useless dimensions. However, if there are no useless dimensions—for example, the Swiss roll—then reducing dimensionality with PCA will lose too much information. You want to unroll the Swiss roll, not squash it.

***51. Suppose you perform PCA on a 1,000-dimensional dataset, setting the explained variance ratio to 95%. How many dimensions will the resulting dataset have?***

That’s a trick question: it depends on the dataset. Let’s look at two extreme examples. First, suppose the dataset is composed of points that are almost perfectly aligned. In this case, PCA can reduce the dataset down to just one dimension while still preserving 95% of the variance. Now imagine that the dataset is composed of perfectly random points, scattered all around the 1,000 dimensions. In this case all 1,000 dimensions are required to preserve 95% of the variance. So the answer is, it depends on the dataset, and it could be any number between 1 and 1,000. Plotting the explained variance as a function of the number of dimensions is one way to get a rough idea of the dataset’s intrinsic dimensionality.

***52. In what cases would you use vanilla PCA, Incremental PCA, Randomized PCA, or Kernel PCA?***

Regular PCA is the default, but it works only if the dataset fits in memory. Incremental PCA is useful for large datasets that don’t fit in memory, but it is slower than regular PCA, so if the dataset fits in memory you should prefer regular PCA. Incremental PCA is also useful for online tasks, when you need to apply PCA on the fly, every time a new instance arrives. Randomized PCA is useful when you want to considerably reduce dimensionality and the dataset fits in memory; in this case, it is much faster than regular PCA. Finally, Kernel PCA is useful for nonlinear datasets.

***53. How can you evaluate the performance of a dimensionality reduction algorithm on your dataset?***

Intuitively, a dimensionality reduction algorithm performs well if it eliminates a lot of dimensions from the dataset without losing too much information. One way to measure this is to apply the reverse transformation and measure the reconstruction error. However, not all dimensionality reduction algorithms provide a reverse transformation. Alternatively, if you are using dimensionality reduction as a preprocessing step before another Machine Learning algorithm (e.g., a Random Forest classifier), then you can simply measure the performance of that second algorithm; if dimensionality reduction did not lose too much information, then the algorithm should perform just as well as when using the original dataset.

***54. Does it make any sense to chain two different dimensionality reduction algorithms?***

It can absolutely make sense to chain two different dimensionality reduction algorithms. A common example is using PCA to quickly get rid of a large number of useless dimensions, then applying another much slower dimensionality reduction algorithm, such as LLE. This two-step approach will likely yield the same performance as using LLE only, but in a fraction of the time.
