# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys

path = "preproc/"

dataIn = pd.read_csv(os.path.join(path, "melodic_mix.csv"),sep = ',',skiprows = 0,header = None) #import data to be segmented
stimulus = pd.read_csv(os.path.join(path, "stimulus.csv")) #import stimuli list
size= len(dataIn)

lf = []
lh = []
rf = []
rh = []
t = []

time = stimulus.Onset

label = stimulus.stim

#separate the stimuli start time by label
for i in range(len(time)):
  if label[i] == "lf":
    lf.append(time[i])
  elif label[i] == "lh":
    lh.append(time[i])
  elif label[i] == "rf":
    rf.append(time[i])
  elif label[i] == "rh":
    rh.append(time[i])
  elif label[i] == "t":
    t.append(time[i])

#transform from time to volume number
for j in range(len(lf)):
  lf[j] = lf[j]/0.72
  lh[j] = lh[j]/0.72
  rf[j] = rf[j]/0.72
  rh[j] = rh[j]/0.72
  t[j] = t[j]/0.72

lf = np.ceil(lf)
lh = np.ceil(lh)
rf = np.ceil(rf)
rh = np.ceil(rh)
t = np.ceil(t)

#initialize matrices
lf_values = np.zeros((len(lf),46))
lh_values = np.zeros((len(lh),46))
rf_values = np.zeros((len(rf),46))
rh_values = np.zeros((len(rh),46))
t_values = np.zeros((len(t),46))

#average 7th, 8th and 9th values after the stimuli starts according to the brain's hemodynamic response
for i in range(46):
  for j in range(len(lf)):
    temp1 = dataIn[i].iloc[int(lf[j]+7)]
    temp2 = dataIn[i].iloc[int(lf[j]+8)]
    temp3 = dataIn[i].iloc[int(lf[j]+9)]
    lf_values[j][i] = (temp1+temp2+temp3)/3

for i in range(46):
  for j in range(len(lh)):
    temp1 = dataIn[i].iloc[int(lh[j]+7)]
    temp2 = dataIn[i].iloc[int(lh[j]+8)]
    temp3 = dataIn[i].iloc[int(lh[j]+9)]
    lh_values[j][i] = (temp1+temp2+temp3)/3

for i in range(46):
  for j in range(len(rf)):
    temp1 = dataIn[i].iloc[int(rf[j]+7)]
    temp2 = dataIn[i].iloc[int(rf[j]+8)]
    temp3 = dataIn[i].iloc[int(rf[j]+9)]
    rf_values[j][i] = (temp1+temp2+temp3)/3

for i in range(46):
  for j in range(len(rh)):
    temp1 = dataIn[i].iloc[int(rh[j]+7)]
    temp2 = dataIn[i].iloc[int(rh[j]+8)]
    temp3 = dataIn[i].iloc[int(rh[j]+9)]
    rh_values[j][i] = (temp1+temp2+temp3)/3

for i in range(46):
  for j in range(len(t)):
    temp1 = dataIn[i].iloc[int(t[j]+7)]
    temp2 = dataIn[i].iloc[int(t[j]+8)]
    temp3 = dataIn[i].iloc[int(t[j]+9)]
    t_values[j][i] = (temp1+temp2+temp3)/3

#data normalization
for i in range(46):
  avg = np.sum(lf_values[:,i])/len(lf_values)
  std = np.std (lf_values[:,i])
  for j in range(len(lf_values)):
    lf_values[j][i] = lf_values[j][i]-avg
    lf_values[j][i] = lf_values[j][i]/std

for i in range(46):
  avg = np.sum(lh_values[:,i])/len(lh_values)
  std = np.std (lh_values[:,i])
  for j in range(len(lh_values)):
    lh_values[j][i] = lh_values[j][i]-avg
    lh_values[j][i] = lh_values[j][i]/std

for i in range(46):
  avg = np.sum(rf_values[:,i])/len(rf_values)
  std = np.std (rf_values[:,i])
  for j in range(len(rf_values)):
    rf_values[j][i] = rf_values[j][i]-avg
    rf_values[j][i] = rf_values[j][i]/std

for i in range(46):
  avg = np.sum(rh_values[:,i])/len(rh_values)
  std = np.std (rh_values[:,i])
  for j in range(len(rh_values)):
    rh_values[j][i] = rh_values[j][i]-avg
    rh_values[j][i] = rh_values[j][i]/std

for i in range(46):
  avg = np.sum(t_values[:,i])/len(t_values)
  std = np.std (t_values[:,i])
  for j in range(len(t_values)):
    t_values[j][i] = t_values[j][i]-avg
    t_values[j][i] = t_values[j][i]/std

#transform data to dataframes for output
lf_pd = pd.DataFrame(lf_values)
lh_pd = pd.DataFrame(lh_values)
rf_pd = pd.DataFrame(rf_values)
rh_pd = pd.DataFrame(rh_values)
t_pd = pd.DataFrame(t_values)

#create output matrix
out = []

for i in range(len(lf_values)):
  out.append([1,0,0,0,0])

for i in range(len(lh_values)):
  out.append([0,1,0,0,0])

for i in range(len(rf_values)):
  out.append([0,0,1,0,0])

for i in range(len(rh_values)):
  out.append([0,0,0,1,0])

for i in range(len(t_values)):
  out.append([0,0,0,0,1])

out_pd = pd.DataFrame(out)

#data output
#input files must be append manually in the order: LF,LH,RF,RH,T in order to obtain new_TrainIn3.csv
lf_pd.to_csv(os.path.join(path, "TrainLF.csv"),header=True,sep=',',index=False)
lh_pd.to_csv(os.path.join(path, "TrainLH.csv"),header=True,sep=',',index=False)
rf_pd.to_csv(os.path.join(path, "TrainRF.csv"),header=True,sep=',',index=False)
rh_pd.to_csv(os.path.join(path, "TrainRH.csv"),header=True,sep=',',index=False)
t_pd.to_csv(os.path.join(path, "TrainT.csv"),header=True,sep=',',index=False)
out_pd.to_csv(os.path.join(path, "new_TrainOut3.csv"),header=True,sep=',',index=False)