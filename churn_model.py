# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 17:59:31 2021

@author: Alexandra
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout

#Importing dataset
dataset = pd.read_csv("Churn_Modelling.csv")

dataset.head()

#Separating independent variables from dependent variables
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:,-1].values

#Converting categorical data to numeric data

#Getting dummy variables
geography = pd.get_dummies(dataset["Geography"], drop_first = True).to_numpy()
gender = pd.get_dummies(dataset["Gender"], drop_first = True).to_numpy()

#Adding the dummy variable columns to original dataset
x = np.concatenate([x, geography, gender], axis = 1)

#Deleting the extra categorical column from original dataset
x = np.delete(x, [1,2], 1)

#Splitting dataset into traiing and test dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 13)


#Scalling columns of training and test dataset
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# %% [markdown]
# # **CREATING AN ARTIFICIAL NEURAL NETWORK**


#Initializing ANN

classifier = Sequential()


#Add the input layer and the first hidden layer

classifier.add(Dense(units = 6, kernel_initializer =  "he_uniform", activation = "relu", input_dim = 11))

#Adding second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", activation = "relu"))

#Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = "glorot_uniform", activation = "sigmoid"))
classifier.summary()


#Compiling the ANN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])


#Fitting the ANN to the Trainig set
model_history = classifier.fit(x_train, y_train, validation_split = 0.33, batch_size = 10, epochs = 100)

#Predicting Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)

#CHECKING MODEL ACCURACY

#Creating confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Checking accuracy score
score = accuracy_score(y_test, y_pred)
print(score)




Based upon data of employees of a bank we calculate whether a employee stands a chance to stay in the company or not.


