# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 15:50:06 2023

@author: david
"""


#FOR Feature Importance With Integrated Gradients
import torch
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader

# Neural Network
from SmallNetwork import SmallNetwork

# Personalized Dataset 
from BaseDataset import BaseDataset

import os
import numpy as np

# Libraries for viewing results
import matplotlib.pyplot as plt

path = os.getcwd() 
pickModel= "models/bestModel073657965474"#INSERT MODEL NAME
pathModel= os.path.join(path, pickModel)

motor_classes = ['LF', 'LH','RF','RH','T']


testSet= BaseDataset("TestIn.csv","TestOut.csv",path)
testloader = DataLoader(testSet, batch_size=1,shuffle=True)  

# Initialize your model, with Correct model specifications (Check results.csv)
model = SmallNetwork(46,5,100,50,20,0.0)

model.load_state_dict(torch.load(pathModel, map_location='cpu'))
model.eval()
# Create an instance of the IntegratedGradients method
ig = IntegratedGradients(model)

# Initialize the feature importance array
num_classes=5
feature_names= [str(i+1) for i in range(46)]

plt.figure(figsize = (9, 9))
# Iterate over the test dataset
for classN in range(num_classes):
  feature_importance = np.zeros([1,46])
  for i, data in enumerate(testloader, 0):
      # Get the input and target
      input, target = data['input'], np.argmax(data['labels'])
      if (target==classN):
        # Compute feature importance for the input of a certain Class
        attributions = ig.attribute(input, target=target)
        feature_importance += attributions.numpy()
      
  # Average the feature importance over the dataset size
  feature_importance /= len(testSet)
  # Create the bar plot
  ax = plt.subplot( num_classes,1, classN+1)
  plt.bar(feature_names, feature_importance.reshape(46))

  # Add title, x-axis label, y-axis label
  title= "Feature Importance for Class = "+motor_classes[classN]
  ax.set_title(title)
  ax.set_xlabel("Number of IC Feature")
  ax.set_ylabel("Importance Value")

  # Add a grid
  ax.grid(True)
  
# Show the plot
plt.tight_layout() 
plt.xlim(0, 47)
plt.show()