from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt

import pandas as pd  
from random import random
import numpy as np



def _load_data(data, n_prev = 20):  
    """
    data should be pd.DataFrame()
    """								

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)
   
    return alsX, alsY 

def train_test_split(df, test_size=0.5):  
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))

    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])	
    return (X_train, y_train), (X_test, y_test)



lines = open('merge_from_ofoct(2).xml','r').read().split('\n')
print lines
lines = filter(None, lines)
flow = [float(i) for i in lines]

flow = np.array(flow).reshape(309,470)
print ("flow\n\n"), flow

data = pd.DataFrame(flow) 

print ("data\n\n"), data
#pdata.b = pdata.b.shift(9)  
#pdata.c = pdata.c.shift(6)
#print ("pdata\n\n"), pdata
#data = pdata.iloc[10:] * random()  # some noise 
#print ("data\n\n"), data

(X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data
print ("x_train ===\n\n"), X_train
print ("y_train ===\n\n"), y_train
print ("x_test ===\n\n"), X_test
print ("y_test ===\n\n"), y_test


in_out_neurons = 470 
hidden_neurons = 500

model = Sequential()  
model.add(LSTM(in_out_neurons, hidden_neurons, return_sequences=False))  
model.add(Dense(hidden_neurons, in_out_neurons))  
model.add(Activation("linear"))  
model.compile(loss="mean_squared_error", optimizer="rmsprop",metrics=['accuracy'])  




# and now train the model
# batch_size should be appropriate to your memory size
# number of epochs should be higher for real world problems
model.fit(X_train, y_train, batch_size=470, nb_epoch=300, validation_split=0.05) 



predicted = model.predict(X_test)  
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))

plt.plot(rmse[:20])
plt.show()


# and maybe plot it
plt.plot(pd.DataFrame(predicted[:20])) 
plt.plot(pd.DataFrame(y_test[:20]))
plt.show()

