# -*- coding: utf-8 -*-

"""**Multinomial (softmax)**"""
import numpy as np

class MulticlassLogisticRegression:
    def __init__(self, lr=0.1, num_iter=5000, fit_intercept=True):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.lambda_value = 0.001 #Regularization term
        self.best_W = None
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __softmax(self, z):
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)
    
    def __loss(self, h, y):
        return (-y * np.log(h)).mean()
    
    def fit(self, X, y, X_val, y_val):
        if self.fit_intercept:
            X = self.__add_intercept(X)
            X_val = self.__add_intercept(X_val)

        self.W = np.zeros((X.shape[1], y.shape[1]))
        
        val_acc=0
        max_acc=0

        for i in range(self.num_iter):
            z = np.dot(X, self.W)
            y_pred = self.__softmax(z)
            regularization_term = self.lambda_value*self.W
            gradient = np.dot(X.T, (y_pred - y)) / y.size + regularization_term
            self.W -= self.lr * gradient
        
            if(i % 2000 == 0):
                z = np.dot(X, self.W)
                y_pred = self.__softmax(z)
                print(f'loss: {self.__loss(y_pred, y)} \t')

            #Validation
            z2 = np.dot(X_val, self.W)
            y_pred_val = self.__softmax(z2)
           
            y_pred_one_hot = np.eye(len(y_pred[0]))[np.argmax(y_pred_val, axis=1)]
            
            val_acc= np.mean(y_pred_one_hot == y_val)
            if max_acc <= val_acc:
              max_acc = val_acc
              max_iteration = i
              self.best_W = self.W

        if (self.best_W.all==self.W.all):
          print("WTF sao iguais?")
        
        print(f"Final W value at row 40:",self.W[40,:])
        print(f"Final best_W value at row 40:",self.best_W[40,:])

        print(f"Iteration where val_acc reached maximum: ",max_iteration)
        print(f"validation accuracy",max_acc)


    def best_predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
            
        return self.__softmax(np.dot(X, self.best_W))

    def best_predict(self, X):
        return np.argmax(self.predict_prob(X), axis=1)

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
            
        return self.__softmax(np.dot(X, self.W))
    
    def predict(self, X):
        return np.argmax(self.predict_prob(X), axis=1)