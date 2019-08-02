# mulight_challenge

Note: I am not reinventing the wheel, most of the information is already spread online. Here I tried to collect, organize, run, test and describe the approach.

Objective:

Two csv files are given, one containing the superhero information and the other superpowers of superheros. 
We are going to first make a feature set and use this set to predict if a superhero is human or not. We are using Pandas and scikit learn.

How to run the code?

The code is initially tested in kaggle notebook and jupyter and then transferred to virtual env. Use the following to install requirements and then run kernel.py,

pip install -r requirements.txt
python kernel.py

Steps:

Data Preparation: Where the data is cleansed and filled in order to have a consistent format for the classifiers. Although random forest is working well with unknown values, in this case we have used a median for height and weight parameters. We dropped the data points without an index and filled the publisher value with the first value in pandas publisher series.

Also the categorical values are converted into indicator values of 0 and 1 using the get_dummies function, for parameters in both info and power files.

Cross fold validation: To split the data into training and test datasets and permutate between them, in this case 5 splits returned better than average result.

Classification: Using in this case, random forest classifier to generate decision trees, train it on training split, and then test them using the voting mechanism of decision trees.

Just as an additional quick try we also used LogisticRegression, from my personal experience LogisticRegression performs well with categorical and low dimensional data. It did generate similar accuracy results to random forest.

Results: For each split calculate the accuracy and get the median in order to have an average for accuracy.

Classifier performance comparison: We generated the confusion matrix in order to see more details on the random forest and logistic regression’s performance (Only with these parameters and without any optimization).

As immediately can be seen from TN, random forest is able to detect more non-humans and also made more FN. LogisticRegression identified more humans and incorrectly also classified humans more than random forest with these parameters. It is hard to judge which classifier is better now without optimization and perhaps more data but in any case interesting to see how they performed and also gives a guideline of how to manipulate the parameters later for our application, based on the sensitivity and accuracy required...


RandomForest confusion matrix: [443,  22],
       [126,  69]



LogisticRegression confusion matrix: [404,  61],
       [ 89, 106]]

