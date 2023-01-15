# -*- coding: utf-8 -*-
"""
One vs Rest Approach (binary logistic regression)
"""
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Binary cross-entropy loss
def binary_cross_entropy(y, y_pred):
  return -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# Logistic regression classifier
class LogisticRegression:
  def __init__(self):
    self.w = None
    self.b = None
    self.best_w = None
    self.best_b = None

  def forward(self, X):
    # Compute the logistic regression prediction
    y_pred = sigmoid(np.dot(X, self.w) + self.b)
    return y_pred   # y_pred.shape = (400,) like 0.43 0.60 ...

  def backward(self, X, y, y_pred, w):
    # Regularization strenght (L2)
    lambda_value = 0.0001
    regularization_term = lambda_value * w
    # Compute the GRADIENTS of the weights and bias  
    dw = np.dot(X.T, y_pred - y) / len(y)  +  regularization_term 
    db = np.sum(y_pred - y) / len(y)
    return dw, db

  def fit(self, X, y,  X_val, y_val, learning_rate, num_iterations):
    # Initialize the weights and bias to 0
    self.w = np.zeros(X.shape[1])
    self.b = 0
    val_acc = []
    max_acc = 0
    current_val_acc = 0

    for i in range(num_iterations):
      # Forward pass
      y_pred = self.forward(X)

      # Backward pass
      dw, db = self.backward(X, y, y_pred,self.w)

      # Update the weights and bias
      self.w -= learning_rate * dw
      self.b -= learning_rate * db

      # Print the cost every 10 iterations
      if i % 5 == 0:
        loss = binary_cross_entropy(y, y_pred).mean()
        #print("Cost at iteration {}: {:.4f}".format(i, loss))   

      # Validation step
      y_pred_val = self.forward(X_val)  # array of probabilities
      y_pred_val = self.predict(y_pred_val) # array binary
          
      current_val_acc = np.mean(y_pred_val == y_val)
      val_acc.append(current_val_acc)

      # We save the best weights, based on the validation set, and use them with 'best_predict_proba' method
      if max_acc<=current_val_acc:
        max_acc=current_val_acc
        max_iteration = i
        self.best_w = self.w
        self.best_b = self.b
 
    print(f"Iteration where val_acc reached maximum: ", max_iteration)
    print(f"validation accuracy",max_acc)


    return self

  def predict(self, X):        
        y_predict = []
        for t in X:
            y_predict.append(1) if (t)>0.5 else y_predict.append(0)
        return np.array(y_predict)    
    
  def predict_proba(self, X):      
        y_predict = []  
        for t in X:        
            y_predict.append(self.forward(t))
        return np.array(y_predict)

  def best_predict_proba(self,X):
        y_predict = [] 
        for t in X:
           y_predict.append(sigmoid(np.dot(t,self.best_w)+self.best_b))
        return np.array(y_predict)

# One-vs-rest classifier
class OvR_LogisticRegression:
  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.model = [LogisticRegression() for _ in range(num_classes)]

  def fit(self, X, y, X_val, y_val, learning_rate,num_iterations):
      
    for i in range(self.num_classes):
      # Train a binary classifier for class i
      
      y_binary = (y == i).astype(int)        
      y_val_binary = (y_val == i).astype(int) 
      self.model[i].fit(X, y_binary, X_val, y_val_binary, learning_rate, num_iterations)
    
    return self

  def predict(self, X):
    y_pred = np.zeros((X.shape[0], self.num_classes))
   
    for i in range(self.num_classes):
      # Predict the probability of class i using the binary classifier
      y_pred[:, i] = self.model[i].predict_proba(X)

  
      y_probabilities = y_pred
      y_classification = np.argmax(y_pred, axis=1)          
      
    # Return the class with the highest probability
    return y_classification, y_probabilities
