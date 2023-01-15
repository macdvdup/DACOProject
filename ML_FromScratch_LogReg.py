# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 18:59:47 2023

@author: david
"""
import numpy as np
import pandas as pd
import os

from OvR_LogisticRegression import OvR_LogisticRegression
from Multinomial_LogisticRegression import Multinomial_LogisticRegression
from BaseDataset import BaseDataset

import matplotlib.pyplot as plt
from auxiliary_results import get_all_roc_coordinates, roc_curve_plot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from sklearn.metrics import roc_auc_score

path = os.getcwd() 

"""# Logistic Regression (from scratch)"""

# Not used

dataIn = pd.read_csv(os.path.join(path, "TrainIn3.csv"),sep = ',',skiprows = 1,header = None)
dataOut = pd.read_csv(os.path.join(path, "TrainOut3.csv"),sep = ',',skiprows = 1,header = None)
dataIn = dataIn.dropna(axis=1)
size= len(dataIn)

# Set training, eval and test percentages
print(size)

propTrain=2/3
# Set training, eval and test 
trainDataIn = dataIn[0:round(propTrain*size)]
evalDataIn = dataIn[round(propTrain*size):]

trainDataOut = dataOut[0:round(propTrain*size)]
evalDataOut = dataOut[round(propTrain*size):]

trainDataIn.to_csv(os.path.join(path, "new_TrainIn.csv"),header=True,sep=',',index=False)
evalDataIn.to_csv(os.path.join(path, "new_EvalIn.csv"),header=True,sep=',',index=False)

trainDataOut.to_csv(os.path.join(path, "new_TrainOut.csv"),header=True,sep=',',index=False)
evalDataOut.to_csv(os.path.join(path, "new_EvalOut.csv"),header=True,sep=',',index=False)
print(trainDataIn.shape)
print(trainDataOut.shape)

# split train data into new_train and new_val

#trainDataIn = dataIn[1:1800]
#evalDataIn = dataIn[1800:2400]
#testDataIn = dataIn[2400:3000]

#trainDataOut = dataOut[1:1800]
#evalDataOut = dataOut[1800:2400]
#testDataOut = dataOut[2400:3000]

#trainDataIn.to_csv(os.path.join(path, "new_TrainIn.csv"),header=True,sep=',',index=False)
#evalDataIn.to_csv(os.path.join(path, "new_EvalIn.csv"),header=True,sep=',',index=False)
#testDataIn = pd.read_csv(os.path.join(path, "TrainIn.csv"),sep=',',skiprows = 1,header = None)

#trainDataOut.to_csv(os.path.join(path, "new_TrainOut.csv"),header=True,sep=',',index=False)
#evalDataOut.to_csv(os.path.join(path, "new_EvalOut.csv"),header=True,sep=',',index=False)
#testDataOut = pd.read_csv(os.path.join(path, "TrainOut.csv"),sep=',',skiprows = 1,header = None)

#print(testDataIn.shape)
#print(testDataOut.shape)

# Do not run 

trainData= BaseDataset("new_TrainIn.csv","new_TrainOut.csv",path)
X_train = trainData.X.to_numpy()
y_train_5d = trainData.y.to_numpy()  # shape (400,5)
y_train_1d = np.argmax(trainData.y.to_numpy(),1) # shape (400, )
valData = BaseDataset("new_EvalIn.csv","new_EvalOut.csv",path)
X_val = valData.X.to_numpy()
y_val_onehot = valData.y.to_numpy()
y_val = np.argmax(y_val_onehot,1)


testData= BaseDataset('TrainIn.csv','TrainOut.csv',path)
X_test = testData.X.to_numpy()
y_test_5d = testData.y.to_numpy()
y_test_1d = np.argmax(testData.y.to_numpy(),1)

output_classes = ('L. Foot', 'L. Hand','R. Foot','R. Hand','Tongue')

"""**Multinomial (softmax)**"""

# Initialize the one-vs-rest classifier
model = MulticlassLogisticRegression()

# Train the classifier on the training data
model.fit(X_train, y_train_5d,X_val,y_val_onehot)

# Evaluate the classifier on the
y_pred = model.predict(X_test)
y_proba = model.predict_prob(X_test)
print(y_proba.shape)
print(y_proba)

# With validation weights:
best_y_pred = model.best_predict(X_test)

acc = np.mean(y_pred == y_test_1d)
acc_val =np.mean(best_y_pred == y_test_1d)
print(f"Accuracy", acc)

print(f"Accuracy with validation", acc_val)


# determine the confusion matrix
confMatrix = confusion_matrix(y_test_1d, y_pred, normalize = None)
display = ConfusionMatrixDisplay(confusion_matrix = confMatrix, display_labels = output_classes)
display= display.plot(cmap=plt.cm.Blues, xticks_rotation=0)
plt.title('LogisticRegression (Multinomial)')

"""**One vs Rest**"""


# Initialize the one-vs-rest classifier
model = OvR_LogisticRegression(5)

# Train the classifier on the training data
model.fit(X_train, y_train_1d, X_val, y_val,learning_rate=0.01,num_iterations=10000)

# Evaluate the classifier on the
y_pred, y_proba = model.predict(X_test)

acc = np.mean(y_pred == y_test_1d)
print(acc)


# determine the confusion matrix
confMatrix = confusion_matrix(y_test_1d, y_pred, normalize = None)
display = ConfusionMatrixDisplay(confusion_matrix = confMatrix, display_labels = output_classes)
display= display.plot(cmap=plt.cm.Blues, xticks_rotation=0)
plt.title('LogisticRegression (One-vs-Rest)')


# Plots the Probability Distributions and the ROC Curves One vs Rest
plt.figure(figsize = (12, 8))
containers = [i/20 for i in range(20)] + [1]
roc_auc_ovr = {}
num_classes=len(output_classes)

for i in range(num_classes):
    # Gets the class
    class_name = output_classes[i]
    y_binary = (y_test_1d == i).astype(int)


    # Prepares an auxiliar dataframe to help with the plots
    testInfo = pd.Series(y_binary).rename("class").to_frame()  #  [   ]
    testInfo['prob'] = y_proba[:, i]  #y_probs
    testInfo = testInfo.reset_index(drop = True)
    
    # Plots the probability distribution for the class and the rest
    ax = plt.subplot(2, num_classes, i+1)
    sns.histplot(data = testInfo, x = 'prob', hue = 'class', color = 'b', ax = ax, bins = containers)
    ax.set_title(class_name)
    ax.legend([f"Class: {class_name}", "Rest"])
    ax.set_xlabel(f"P(Class = {class_name})")
    
    # Calculates the ROC Coordinates and plots the ROC Curves
    ax_bottom = plt.subplot(2, num_classes, i+1+num_classes)
    tpr, fpr = get_all_roc_coordinates(testInfo['class'], testInfo['prob'])  # 
    roc_curve_plot(tpr, fpr, scatter = False, ax = ax_bottom)
    ax_bottom.set_title(f"ROC Curve - {class_name} vs R")
    
 
    # Calculates the ROC AUC OvR
    roc_auc_ovr[class_name] = roc_auc_score(testInfo['class'], testInfo['prob'])
plt.tight_layout()