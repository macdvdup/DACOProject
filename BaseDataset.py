# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 03:30:23 2023

@author: david
"""
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

# Dataset
class BaseDataset(Dataset):
    def __init__ (self, csv_fileIn,csv_fileOut, root_dir,isTest=False):
        motor_classes = ['LF', 'LH','RF','RH','T']
        self.infoInput = pd.read_csv(os.path.join(root_dir, csv_fileIn),sep = ',',skiprows = 1,header = None)
        self.infoOutput = pd.read_csv(os.path.join(root_dir, csv_fileOut),sep = ',',skiprows = 1,header = None, names=motor_classes)
        self.root_dir = root_dir
        self.infoInput = self.infoInput.dropna(axis=1)
        allTheData = pd.concat([self.infoInput, self.infoOutput], axis=1, join="inner")
        allTheData = allTheData.sample(frac = 1)
        
        self.infoOutput=allTheData[[x for x in motor_classes]]
        self.infoInput=allTheData[[i for i in range(46)]]
        #if(~isTest):
          #self.infoInput=(self.infoInput-self.infoInput.mean())/self.infoInput.std()
    def __len__ (self):
        return len(self.infoInput)
    
    def __getitem__ (self, index):
        # gets the important information about each image: its name and label
        input = self.infoInput.iloc[index].to_numpy().astype(np.float32)
        labels = self.infoOutput.iloc[index].to_numpy().astype(np.float32)
              
        sample = {'input': input, 'labels': labels}
        return sample