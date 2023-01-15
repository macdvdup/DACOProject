# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 18:59:47 2023

@author: david
"""
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset

from OvR_LogisticRegression import OvR_LogisticRegression, BinaryLogisticRegression
from Multinomial_LogisticRegression import MulticlassLogisticRegression
from BaseDataset import BaseDataset

import matplotlib.pyplot as plt
from auxiliary_results import get_all_roc_coordinates, roc_curve_plot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from sklearn.metrics import roc_auc_score

path = os.getcwd() 

"""#

 Small changes on the BaseDataset

 Reading csv files

"""

# Dataset
class BaseDataset(Dataset):
    def __init__ (self, csv_fileIn,csv_fileOut, root_dir,isTest=False):
        self.X = pd.read_csv(os.path.join(root_dir, csv_fileIn),sep = ',',skiprows = 1,header = None)
        self.y = pd.read_csv(os.path.join(root_dir, csv_fileOut),sep = ',',skiprows = 1,header = None)
        self.root_dir = root_dir
        
    def __len__ (self):
        return len(self.X)
    
    def __getitem__ (self, index):
        # gets the important information about each image: its name and label
        input = self.X.iloc[index].to_numpy().astype(np.float32)
        labels = self.y.iloc[index].to_numpy().astype(np.float32)
              
        sample = {'input': input, 'labels': labels}
        return sample


trainData= BaseDataset("new_TrainIn3.csv","new_TrainOut3.csv",path)
X_train = trainData.X.to_numpy() # shape (1000, 47)
y_train_5d = trainData.y.to_numpy()  # shape (1000,5)
y_train_1d = np.argmax(trainData.y.to_numpy(),1) # shape (1000, )

valData = BaseDataset("EvalIn3.csv","EvalOut3.csv",path)
X_val = valData.X.to_numpy()
y_val_onehot = valData.y.to_numpy()
y_val = np.argmax(y_val_onehot,1)

testData= BaseDataset('TestIn3.csv','TestOut3.csv',path)
X_test = testData.X.to_numpy()
y_test_5d = testData.y.to_numpy()
y_test_1d = np.argmax(testData.y.to_numpy(),1)

output_classes = ('L. Foot', 'L. Hand','R. Foot','R. Hand','Tongue')

"""# Logistic Regression (from scratch)"""

"""**Multinomial (softmax) Approach**"""

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
print(f"Multinomial (softmax) Approach Accuracy ", acc)

print(f"Accuracy with validation", acc_val)


# determine the confusion matrix
confMatrix = confusion_matrix(y_test_1d, y_pred, normalize = None)
display = ConfusionMatrixDisplay(confusion_matrix = confMatrix, display_labels = output_classes)
display= display.plot(cmap=plt.cm.Blues, xticks_rotation=0)
plt.title('LogisticRegression (Multinomial)')



"""**One vs Rest Approach**"""


# Initialize the one-vs-rest classifier
model = OvR_LogisticRegression(5)

# Train the classifier on the training data
model.fit(X_train, y_train_1d, X_val, y_val,learning_rate=0.01,num_iterations=10000)

# Evaluate the classifier on the
y_pred_2, y_proba_2 = model.predict(X_test)

acc_2 = np.mean(y_pred == y_test_1d)
print(f"One vs Rest Approach Accuracy", acc_2)


# determine the confusion matrix
confMatrix = confusion_matrix(y_test_1d, y_pred_2, normalize = None)
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
    testInfo = pd.Series(y_binary).rename("class").to_frame() 

    # Use 'y_proba_2' if we want the One-vs-Rest ROC-AUC !!
    testInfo['prob'] = y_proba[:, i]  
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