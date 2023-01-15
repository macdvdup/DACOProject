# -*- coding: utf-8 -*-
# Torch Imports
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Personalized Dataset 
from BaseDataset import BaseDataset

# Neural Network
from SmallNetwork import SmallNetwork

# General Imports
import tqdm
import numpy as np
import pandas as pd
from datetime import datetime

# Libraries for viewing results
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from auxiliary_results import get_all_roc_coordinates, roc_curve_plot
import seaborn as sns
from sklearn.metrics import roc_auc_score

import os # For paths to save data

'''
FOR GOOGLE COLLAB
from google.colab import drive 
drive.mount('/gdrive')'''

# Sets a unique time to give to 
current_dateTime = str(datetime.now()).split(" ")[1]
current_dateTime=current_dateTime.replace('.', '').replace(':', '')

#Set Paths
path = os.getcwd() 
pathToSave = os.path.join(path, "figures")
pathModel= os.path.join(path, "models/bestModel"+current_dateTime)

# Training, Validation and Test Sets
trainSet= BaseDataset("new_TrainIn3.csv","new_TrainOut3.csv",path) 
evalSet = BaseDataset("EvalIn3.csv","EvalOut3.csv",path) 

# Get number of input Features and number of classes (46 and 5)
num_input_features= trainSet.infoInput.shape[1]
num_classes= trainSet.infoOutput.shape[1]

# Set number of nodes in hiddenLayer1 and hiddenLayer2, and the dropoutRate,
fc1=100
fc2=50
dropoutRate=0.0
weight_decay=5E-4

# Set Model
model=SmallNetwork(num_input_features,num_classes,fc1,fc2,dropoutRate)
motor_classes = ('LF', 'LH','RF','RH','T')

# Set optimizer and Criterion for 
learning_rate=0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) #
criterion= nn.CrossEntropyLoss()

batch_size=len(trainSet)
trainloader = DataLoader(trainSet, batch_size=batch_size)
evaloader = DataLoader(evalSet, batch_size=batch_size)

num_epochs=1000
trainingAccuracy=[]
evalAccuracy=[]
trainingLoss=[]
epochList=[]
epochMax=0
maxAcc=0
for epoch in tqdm.tqdm(range(num_epochs)):
  # Initializing variables and append epoch number
  epochList.append(epoch)  
  correct=0
  total=0

  model.train()
  #Training
  for i, data in enumerate(trainloader, 0):
    input,label = data['input'], data['labels']

    output = model(input)
    
    loss = criterion(output, label.double())
    loss.backward()
    optimizer.step()
    
    _, predicted = torch.max(output.data, 1)
    _,labelClass = torch.max(label, 1)
    
    correct += (predicted == labelClass).sum().item()
    total += labelClass.size(0)

  #Training Data
  trainAcc = 100 * correct / total
  trainingAccuracy.append(trainAcc)

  # Evaluation    
  correct=0
  total=0  
  
  model.eval()
  with torch.no_grad():
    for i, data in enumerate(evaloader, 0):
      input,label = data['input'], data['labels']
      output = model(input)
      _, predicted = torch.max(output.data, 1)
      _, labelClass = torch.max(label, 1)
      
      correct += (predicted == labelClass).sum().item()
      total += labelClass.size(0)

  #Evaluation Data
  evalAcc = 100 * correct / total
  evalAccuracy.append(evalAcc)
  if maxAcc<=evalAcc:
      maxAcc=evalAcc
      epochMax=epoch
      torch.save(model.state_dict(), pathModel)

print('Best Model Accuracy in Eval Set: ', maxAcc, ' %')

# PLOTS OF Training And Eval Accuracy
fig1, ax1 = plt.subplots(figsize = (20, 6))
ax1.plot(np.array(epochList), np.array(trainingAccuracy), '--', color = 'blue')

figure_name = 'trainAcc2'+current_dateTime 
figure_path = os.path.join(pathToSave, f"{figure_name}")
plt.savefig(fname = figure_path, bbox_inches = 'tight')

fig2, ax2 = plt.subplots(figsize = (20, 6))
ax2.plot(np.array(epochList), np.array(evalAccuracy), '--', color = 'red')
ax2.plot(np.array(epochMax), np.array(maxAcc), 'o', color = 'green')

figure_name = 'evalAcc2'+current_dateTime 
figure_path = os.path.join(pathToSave, f"{figure_name}")
plt.savefig(fname = figure_path, bbox_inches = 'tight')

# TESTING
testSet= BaseDataset("TestIn.csv","TestOut.csv",path)
testloader = DataLoader(testSet, shuffle=True)  

correct=0
acc=0
total=0
testOutputs = np.array([])
testLabels = np.array([])
testProbs = np.zeros(shape=(1,5))
testLabelsArrays = np.zeros(shape=(1,5))

modelToTest = SmallNetwork(num_input_features,num_classes,fc1,fc2,dropoutRate)
modelToTest =  modelToTest.load_state_dict(torch.load(pathModel))
with torch.no_grad():
  
  for i, data in enumerate(testloader, 0):
    
    input,label = data['input'], data['labels']
    output = model(input)
    _, predicted = torch.max(output.data, 1)
    _, labelClass = torch.max(label, 1)

    testOutputs = np.concatenate((testOutputs, predicted), axis=None)
    testLabels = np.concatenate((testLabels, labelClass), axis=None)
    testProbs = np.concatenate((testProbs,output.numpy()), axis=0)
    testLabelsArrays = np.concatenate((testLabelsArrays,label.numpy()), axis=0)

    correct += (predicted == labelClass).sum().item()
    total += labelClass.size(0)

testProbs = testProbs[1:,:]
testLabelsArrays = testLabelsArrays[1:,:]
testAcc = 100 * correct / total

print('Best Model Accuracy in Test Set: ', testAcc, ' %')


# determine the confusion matrix
motor_classes = ('LF', 'LH','RF','RH','T')

confMatrix = confusion_matrix(testLabels, testOutputs, normalize = None)
display = ConfusionMatrixDisplay(confusion_matrix = confMatrix, display_labels = motor_classes)
display= display.plot(cmap=plt.cm.Blues, xticks_rotation=0)
plt.title('Confusion Matrix - Small Network')

figure_name = 'confMat2'+current_dateTime 
figure_path = os.path.join(pathToSave, f"{figure_name}")
plt.savefig(fname = figure_path, bbox_inches = 'tight')


# Plots the Probability Distributions and the ROC Curves One vs Rest
plt.figure(figsize = (12, 8))
containers = [i/20 for i in range(20)] + [1]
roc_auc_ovr = {}

for i in range(num_classes):
    # Gets the class
    class_name = motor_classes[i]
    
    # Prepares an auxiliar dataframe to help with the plots
    testInfo = pd.Series((testLabelsArrays[:,i]==1)*1).rename("class").to_frame()  #  [   ]
    testInfo['prob'] = testProbs[:, i]  #y_probs
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
    ax_bottom.set_title(f"ROC Curve {class_name}vR")
    
    # Calculates the ROC AUC OvR
    roc_auc_ovr[class_name] = roc_auc_score(testInfo['class'], testInfo['prob'])

plt.tight_layout()
plt.tripcolor

figure_name = 'plotsAUC2'+current_dateTime 
figure_path = os.path.join(pathToSave, f"{figure_name}")
plt.savefig(fname = figure_path, bbox_inches = 'tight')

optName= type (optimizer).__name__
dataToSave={"model": str(model),"test_acc":testAcc,"learning_rate":learning_rate,"dropout_Rate":model.dropout.p,"batch_size":batch_size,"weight_decay":weight_decay,"optimizer":optName}
moreData={}
for classN in motor_classes:
  moreData["ROC-AUC"+classN]=roc_auc_ovr[classN]
dataToSave.update(moreData)
dataToSave = pd.DataFrame([dataToSave])
dataToSave.to_csv(os.path.join(pathToSave,"results.csv"), mode='a', index=False, header=False)