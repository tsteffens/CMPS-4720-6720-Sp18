
# coding: utf-8

# In[497]:


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


# In[498]:


#fetch data from quandl
start = dt.datetime(1995,1,1)
end = dt.date.today()
df = web.DataReader('CAT','quandl',start,end)

#select only adjusted Open,High,Low,Close columns
df.drop(df.columns[[0,1,2,3,4,5,6,8,9,11]], axis = 1, inplace =True)
df.head()



# In[499]:


#Normalize data by dividing by dividing by 10000
df['AdjOpen'] = df['AdjOpen']/10000
#df['AdjHigh'] = df['AdjHigh']/10000
df['AdjClose'] = df['AdjClose']/10000
#df['AdjLow'] = df['AdjLow']/10000
df.head()


# In[513]:


def load_data(ticker, sLen):
    numfeatures = len(ticker.columns)
    data = ticker.as_matrix() #pd.DataFrame(stock)
    seqLength = sLen + 1
    frame = []
    for index in range(len(data) - seqLength):
        frame.append(data[index: index + seqLength])

    frame = np.array(frame)
    
    ############determine train vs test ratio#############
    row = round(0.95 * frame.shape[0])
    #######################################################
    
    train = frame[:int(row), :]
    X_train = train[1:, :-1]
    y_train = train[:-1, -1][:,-1]
    X_test = frame[int(row):, :-1]
    y_test = frame[int(row):, -1][:,-1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], numfeatures))
    x_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], numfeatures))  

    return [X_train, y_train, X_test, y_test]


# In[514]:


def make_model(layers):
        money_maker = Sequential()
        
        #first layer of lstm
        money_maker.add(LSTM(128,
                       input_shape=(layers[1],
                                    layers[0]),
                       return_sequences=True))
        
        #dropout of 20%
        money_maker.add(Dropout(.2))
        
        #2nd layer of lstm
        money_maker.add(LSTM(64,
                       input_shape=(layers[1], 
                                    layers[0]),
                       return_sequences=False))
        
        #dropout of 20%
        money_maker.add(Dropout(.2))
        
        #fully connected layer
        money_maker.add(Dense(16,
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


# In[515]:


#print size of x,y test and train

##########prediction length#############
pl = 5
################################################################

X_train, y_train, X_test, y_test = load_data(df[::-1], pl)
print("X train", X_train.shape)
print("y train", y_train.shape)
print("X test", X_test.shape)
print("y test", y_test.shape)


# In[516]:


model = make_model([2,pl,1])


# In[517]:


model.fit(
    X_train,
    y_train,
    batch_size=512,
    
    ##########Number of Epochs############
    epochs=100,
    #######################################
    validation_split=0.1,
    verbose=1)


# In[519]:


train_eval = model.evaluate(X_train, y_train, verbose=1)
print(train_eval)
print('Train MSE: %f MSE' % (trainScore[0],))

test_eval = model.evaluate(X_test, y_test, verbose=1)
print(test_eval)
print('Test MSE: %f MSE' % (testScore[0]))


# In[520]:


error=[]
prediction = model.predict(X_test)
for i in range(len(prediction)):
    prediction[i][0] = prediction[i][0]*10000
y_test_act = y_test*10000

for i in range(len(y_test)):
    error.append((y_test_act[i]-prediction[i][0])**2)
totalErr = sum(error)
mse = totalErr/len(y_test)
print(mse)
    


# In[521]:


import matplotlib.pyplot as plt

plt.plot(prediction,color='red', label='prediction')
plt.plot(y_test_act,color='blue', label='y_test')
plt.legend(loc='lower right')
plt.show()

print(model.predict(X_test))

