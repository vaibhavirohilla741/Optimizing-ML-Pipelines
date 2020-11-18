# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

![alt text](https://github.com/vaibhavirohilla741/Optimizing-ML-Pipelines/blob/main/img/architecture.png "Logo Title Text 1")

## Summary
This dataset contains data about the phonecall to a bank.It consist of data columns like age,marital status, job, education etc.We seek to predict that the user will take the term deposite.

There were two different ways in which we have applied the model firstly by using scikit learn to fit the model and then by using the hypertunning we tunned the model.Next method was the automl model and lastly we compared the result of two.

## The accuracy which we got from the two models were
AutoML Accuracy 0.9179059180576631<br>
Hyperdrive Accuracy 0.9072837632776934

## Scikit-learn Pipeline
We have been provided a train.py file for preparing the data and to use it for our further models.
Fistly we have loaded the dataset using the tabular dataset factory dataset.
next task was to clean the data which is a mojor part in any ml model.
next task was to split the data into train and test dataset.
the final step was to apply the scikit learn model to fit the train data and find out the accuracy of the model.
lastly we save the final model.

The classification algorithm we have used was Logistic Regression.It is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables.

We choose a random parameter sampling strategy as it provides the ranges for the learning rate and the dropout hyperparameters in the RandomParameterSampling object.There are huge benifits of using this such as it is very easy to use and it offers an unbiased selection and it is highly representative, it also enables researches to get clear conclusions.

An estimator that will be called with sampled hyperparameters:  SKLearn creates an estimator for training in Scikit-learn experiments (logistic regression model is importet from Scikit-learn); here we also specify the compute target to be used.

Primary metric name and goal: The name of the primary metric reported by the experiment runs (accuracy) and if we wish to maximize or minimize the primary metric (maximize).

An early termination policy specifies that if you have a certain number of failures, HyperDrive will stop looking for the answer.
We have defined an early termination policy to be used as BanditPolicy.This will terminate jobs that are not performing well and are not likely to yield a good model in the end.

lastly we collect and save the best model, that is, logistic regression with the tuned hyperparameters which yield the best accuracy score.

## AutoML
In the automl model ie. <u>Pre-fitted Soft Voting Classifier class model</u> we have to again load the dataset using the tabular dataset factory dataset.
we have initilise the automl configration and defined experiment_timeout_minutes, task, primary_metric, training_data, label_column_name, compute_target.
lastly we run the automl and collected the best accuracy model.<br>
AutoMl parameters are as follows

* RawFeatureName: feature or column name present in the dataset.<br>
* TypeDetected: input datatype<br>
* Dropped: Indicates if the input feature was dropped or used,<br>
* EngineeringFeatureCount: Number of features generated through automated feature engineering transforms,<br>
* Transformations: List of transformations applied to input features to generate engineered features.<br>

![alt text](https://github.com/vaibhavirohilla741/Optimizing-ML-Pipelines/blob/main/img/automl.JPG)

## Hyperparameters used in AutoMl
 min_split_gain=0.47368421052631576,<br>
 n_estimators=600,<br>
 n_jobs=1,<br>
 num_leaves=131,<br>
 objective=None,<br>
 random_state=None,<br>
 reg_alpha=0.05263157894736842,<br>
 reg_lambda=0.05263157894736842,<br>
 silent=True,<br>
 subsample=1,<br>
 subsample_for_bin=200000,<br>
 subsample_freq=0,<br>
 verbose=-10))],<br>
 verbose=False))],<br>
 flatten_transform=None,<br>
 weights=[0.125, 0.125, 0.25,0.125, 0.125, 0.125,0.125]))],<br>
 verbose=False)<br>

## Comparison between the two
the two models were mostly similar as we have to load and prepare the dataset in both the dataset. But if to choose one the automl is much better as we do not have to do much work and code there and the result ie. the acuracy we got is nearly similar.
In the hyperdrive model we have to define many thing such as parameters and early stopping policy and primary metric to optimize.

In my understanding the automl model is mmuch better as compared to hyperdrive. 

in hyperdrive we need to do the following things before the final hyperdrive run.

+ Define the parameter search space<br>
+ Specify a primary metric to optimize<br>
+ Specify early termination policy for low-performing runs<br>
+ Allocate resources<br>
+ Launch an experiment with the defined configuration<br>
+ Visualize the training runs<br>
+ Select the best configuration for your model<br>
+ In AutomMl all these things are done automatically so it is time saving and efficient way.<br>
## Future work
+ We can try diffrent model and classification algorithm to get better result.
+ Grid and Grid sampling can also be used in the hyper drive model
+ Light GBM -A fast, distributed, high-performance gradient-boosting (GBDT, GBRT, GBM, or MART) framework based on decision tree algorithms. It's used for ranking, classification, and many other machine-learning tasks.<br>
+ XGBoost- A fast, portable, and distributed gradient-boosting (GBDT, GBRT, or GBM) library for Python, R, Java, Scala, C++, and more. It runs on a single machine, and on Apache Hadoop and Spark.


