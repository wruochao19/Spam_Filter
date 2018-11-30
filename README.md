# Spam_Filter
This project is a spam filter module with Machine Learning based on Python using Bayes. This filter use Classic Naive Bayes to classify given mails basing on wether they are spam or not.

This Spam filter use dataset from kaggle to train and test. the dataset contains 5573 email, among them 13% is spam and rest of them is healthy.

We do two parts,the first part implements Navie Bayes algorithm by our own code. Second part uses third part library sklearn to do spam filter.
# Naive Bayes
In machine learning, naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
Abstractly, naive Bayes is a conditional probability model: given a problem instance to be classified, represented by a vector 
![Aaron Swartz](https://github.com/wruochao19/Hello-world/raw/master/1.png)
 
 representing some n features (independent variables), it assigns to this instance probabilities 
![Aaron Swartz](https://github.com/wruochao19/Hello-world/raw/master/2.png)

The problem with the above formulation is that if the number of features n is large or if a feature can take on a large number of values, then basing such a model on probability tables is infeasible. We therefore reformulate the model to make it more tractable. Using Bayes' theorem, the conditional probability can be decomposed as 
![Aaron Swartz](https://github.com/wruochao19/Hello-world/raw/master/4.png)

**project tree**
----------------
**Naive_Bayes**
 * Naive_Bayes.py ``collect feature and lable, and classify email.``
 * train.py ``using dataset to train classify moudle.``
 * test.py ``classify a give number of mails and output the error rate``
**sk_leaern_based**
 * sk_learn_based.py ``import sklearn library to split data, train model, evaluate, filter new``
----------------
**Dependency List**
 * numpy
 * panda
 * re
 * sklearn
----------------
# result:

**Simple Naive Bayes**
-----------------
 * Using 5574 traing spam. Classfiying 1000 mails,By our own implementation the average error rate is 0.9% 
-----------------
**sklearn**
 * while the sklearn gets a 0.5% averagely error rate.
