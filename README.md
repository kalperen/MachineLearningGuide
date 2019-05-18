# MachineLearningGuide

What is Machine Learning:

(Tom Mitchell) Well-Posed Learning Problem: A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.

Supervised Learning:
We are given a data set and know what the correct output should look like for each input.
Supervised Learning problems can be "regression" or "classification" problems.
Regression problems consist of trying to predict results within a continuous output. i.e: Given a picture of a person, we have to predict their age on the basis of the given picture
Classification problems consists of trying to predict results in a discrete output. i.e: Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.

Unsupervised Learning:
Derive structure from data where we don't necessarily know the effect of the variables.

Clustering: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.

Non-clustering: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a cocktail party).

Model Representation and definitions.

x^{(i)}: “input” variables, also called input features
y^{(i)}: “output” or target variable that we are trying to predict.
A pair (x^{(i)} , y^{(i)} )is called a training example, and the dataset that we’ll be using to learn—a list of m training examples (x(i),y(i));i=1,...,m—is called a training set.
We will also use X to denote the space of input values, and Y to denote the space of output values.

Hypothesis function: In Supervised learning the goal is, given a training set, to learn a function h : X → Y so that h(x) is a “good” predictor for the corresponding value of y.

Cost Function:
![Alt text](/images/formulas/costFunction.png?raw=true "Cost Function")
