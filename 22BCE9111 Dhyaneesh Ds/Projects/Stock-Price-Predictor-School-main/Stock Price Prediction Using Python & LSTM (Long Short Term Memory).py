#!/usr/bin/env python
# coding: utf-8

# In[284]:


#IMPORTING LIBRARIES 
import math
import numpy as np
import pandas as pd 
import pandas_datareader
import matplotlib.pyplot as pyplot

#A sequential model takes into account the order of things happening and is linear, it is also fairly simple to use.
from keras.models import Sequential

#Dense layer is the regular deeply connected neural network layer. Essentially a vanilla NN
#The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Input data may have some of the unwanted data, usually called as Noise. Dropout will try to remove the noise data and thus prevent the model from over-fitting.
#LSTM is a Recurrent neural network that helps in sequential data (Data that needs to preserve order) and also helps fight the forgetting problem in vanilla RNN 
from keras.layers import LSTM,Dropout,Dense

#Transforms any actual range into a range of 0 to 1 so it is easier to work with, essentially the Sigmoid function / ReLU
from sklearn.preprocessing import MinMaxScaler


# In[343]:


#IMPORTING DATA
df = pandas_datareader.DataReader('AAPL',data_source='yahoo', start= "2012-01-01", end="2019-12-31" )
df


# In[344]:


#VISUALISING CLOSING PRICES

#using my prefered styling for our graphs
pyplot.style.use('dark_background')
#Plotting the graph
pyplot.figure(figsize=(20,20))
pyplot.plot(df['Close'])
pyplot.title("Closing Prices History from 2012-2019")
pyplot.xlabel('Date', fontsize=20)
pyplot.ylabel('Close Price USD ($)', fontsize=20)
pyplot.show()


# In[345]:


data = df.filter(['Close'])
#convert data frame to be numpy compatible 
dataset = data.values

#The rest is used to test 
train_len = math.ceil(len(dataset) * 0.8)


# In[346]:


#Scaling the data and Creating training data sets 
#Converting the data from arrays of hundreds to arrays of 0 to 1s so we can work with them efficiently 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
#Creating the training data set 
train_data = scaled_data[0:train_len, :]

#x is the independant variable and y is dependant
x_train,y_train=[],[]

for i in range(60,len(train_data)):
    #We are leaving out some of the data set to test our model later
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
    
#Converting to numpy arrays / tensors to be used in models
x_train,y_train=np.array(x_train),np.array(y_train)
#We Reshape the model as LSTM requires a 3d shape so we just insert 1 to the 3rd dimension
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


# In[347]:


#Build the LSTM model 
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compilig the Model 
#A loss / cost function tells the computer how wrong it was from the actual result and a optimizer acts as a catalyst
model.compile(loss='mean_squared_error',optimizer='adam')

#Training the model
model.fit(x_train,y_train,batch_size=1,epochs=1)
#batch_size = how many batches at a time 
#epochs = how many times to run this program


# In[348]:


#Creating Testing Data
test_data = scaled_data[train_len - 60: , :]
#Create data sets 
test_set = []
actual_values = dataset[train_len:,:]

#Using Rest of the dataframe
for i in range(60,len(test_data)):
    test_set.append(test_data[i-60:i,0])
    
#Converting test_set to numpy array and reshaping it
test_set = np.array(test_set)
test_set = np.reshape(test_set,(test_set.shape[0],test_set.shape[1],1))


# In[349]:


#Get predicted prices
predicted_closing_price = model.predict(test_set)
predicted_closing_price = scaler.inverse_transform(predicted_closing_price)


# In[350]:


#Getting standard deviation / RMSE (Root Mean Squared Error) it is a way to calculate the accuracy of our model
rmse =np.sqrt(np.mean(((predicted_closing_price- actual_values)**2)))
rmse = np.sqrt(np.mean(np.power((np.array(actual_values)-np.array(predicted_closing_price)),2)))
rmse = np.sqrt(((predicted_closing_price - actual_values) ** 2).mean())
rmse =  round(rmse,2)


# In[352]:


#Plot the data 
train = data[:train_len]
valid = data[train_len:]
valid['Predictions'] = predicted_closing_price
#Visualisation
pyplot.figure(figsize=(20,20))
pyplot.title("Closing Prices History from 2012-2019")
pyplot.xlabel('Date', fontsize=20)
pyplot.ylabel('Close Price USD ($)', fontsize=20)
pyplot.plot(train['Close'])
pyplot.plot(valid[['Close','Predictions']])
pyplot.legend(['Training Set','Actual Values','Predictions'],loc="lower right")
pyplot.text(15250,75,f'RMSE of given prediction is {rmse}', fontsize = 22)
pyplot.show()

