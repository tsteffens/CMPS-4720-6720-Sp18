
# coding: utf-8

# In[186]:


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import pandas_datareader.data as web
import datetime as dt
from sklearn import preprocessing
import datetime
from sklearn.metrics import mean_squared_error


# In[187]:


#fetch data from quandl
start = dt.datetime(1995,1,1)
end = dt.date.today()
df1 = web.DataReader('CAT','quandl',start,end)
df2 = web.DataReader('DE','quandl',start,end)
df3 = web.DataReader('CMI','quandl',start,end)
df4 = web.DataReader('TEX','quandl',start,end)


#select only adjusted close
df1.drop(df1.columns[[0,1,2,3,4,5,6,7,8,9,11]], axis = 1, inplace =True)
df2.drop(df2.columns[[0,1,2,3,4,5,6,7,8,9,11]], axis = 1, inplace =True)
df3.drop(df3.columns[[0,1,2,3,4,5,6,7,8,9,11]], axis = 1, inplace =True)
df4.drop(df4.columns[[0,1,2,3,4,5,6,7,8,9,11]], axis = 1, inplace =True)


df1.rename(columns = {'AdjClose': 'CAT'}, inplace=True)
df2.rename(columns = {'AdjClose': 'DE'}, inplace=True)
df3.rename(columns = {'AdjClose': 'CMI'}, inplace=True)
df4.rename(columns = {'AdjClose': 'TEX'}, inplace=True)

#join tickers into main dataframe
main_df = pd.DataFrame()
main_df = main_df.join(df2, how='outer')
main_df = main_df.join(df3, how='outer')
main_df = main_df.join(df4, how='outer')
main_df = main_df.join(df1, how='outer')
print(main_df.head())




# In[188]:


#normalize data
main_df['CAT'] = main_df['CAT']/1000
main_df['DE'] = main_df['DE']/1000
main_df['CMI'] = main_df['CMI']/1000
main_df['TEX'] = main_df['TEX']/1000
main_df.head()


# In[189]:


def load_data(ticker, sLen):
    seqLength = sLen + 1
    frame = []
    
    numRelated = len(ticker.columns)
    data = ticker.as_matrix()
    for i in range(len(data) - seqLength):
        frame.append(data[i: i + seqLength])

    frame = np.array(frame)
    
    ############determine train vs test ratio#############
    split = round(0.95 * frame.shape[0])
    #######################################################
    
    train = frame[:int(split), :]
    X_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    X_test = frame[int(split):, :-1]
    y_test = frame[int(split):, -1][:,-1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], numRelated))
    x_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], numRelated))  

    return [X_train, y_train, X_test, y_test]


# In[190]:


def make_model(layers):
        money_maker = Sequential()
        
        #first layer of lstm
        money_maker.add(LSTM(120,
                       input_shape=(layers[1],
                                    layers[0]),
                       return_sequences=True))
        
        #dropout of 20%
        money_maker.add(Dropout(.2))
        
        #2nd layer of lstm
        money_maker.add(LSTM(60,
                       input_shape=(layers[1], 
                                    layers[0]),
                       return_sequences=False))
        
        #dropout of 20%
        money_maker.add(Dropout(.2))
        
        #fully connected layer
        money_maker.add(Dense(15,
                        kernel_initializer='uniform',
                        activation='relu'))   
        #fully connected layer
        money_maker.add(Dense(1,
                        kernel_initializer='uniform',
                        activation='relu'))
        
        #compile model using loss as meansquareerror 
        #and adam(stochastic gradient descent) as optimizer
        #learning rate =.001 
        money_maker.compile(loss='mse',optimizer='adam', metrics = ['mse','mape'])
        return money_maker


# In[191]:


#print size of x,y test and train

##########prediction length#############
pl = 5
################################################################

X_train, y_train, X_test, y_test = load_data(main_df[::-1], pl)
print("X train", X_train.shape)
print("y train", y_train.shape)
print("X test", X_test.shape)
print("y test", y_test.shape)


# In[192]:


model = make_model([4,pl,1])


# In[193]:


model.fit(
    X_train,
    y_train,
    #max batch_size = GPU mem/4/(size of tensor + trainable params)
    batch_size=500,
    
    ##########Number of Epochs############
    epochs=500,
    #######################################
    validation_split=0.1,
    verbose=1)


# In[194]:


train_eval = model.evaluate(X_train, y_train, verbose=1)
print(train_eval)
print('Train MSE: %f MSE' % (train_eval[0],))

test_eval = model.evaluate(X_test, y_test, verbose=1)
print(test_eval)
print('Test MSE: %f MSE' % (test_eval[0]))


# In[195]:


#calculate unnormalized MSE
error=[]
prediction = model.predict(X_test)
for i in range(len(prediction)):
    prediction[i][0] = prediction[i][0]*1000
y_test_act = y_test*1000

for i in range(len(y_test)):
    error.append((y_test_act[i]-prediction[i][0])**2)
totalErr = sum(error)
mse = totalErr/len(y_test)
print(mse)
    


# In[196]:


import matplotlib.pyplot as plt

plt.plot(prediction,color='red', label='prediction')
plt.plot(y_test_act,color='blue', label='y_test')
plt.legend(loc='lower right')
plt.show()

