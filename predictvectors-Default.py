from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt

import pandas as pd  
from random import random
import numpy as np


def _load_data(data, n_prev = 100):  
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


flow = (list(range(1,10,1)) + list(range(10,1,-1)))*30
print ("flow\n\n"), len(flow)
pdata = pd.DataFrame({"a":flow, "b":flow,"c":flow }) 
print ("pdata\n\n"), pdata
pdata.b = pdata.b.shift(9)  
pdata.c = pdata.c.shift(6)
print ("pdata\n\n"), pdata
data = pdata.iloc[10:] * random()  # some noise  
print ("data\n\n"), data


(X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data
print ("x_train ===\n\n"), X_train
print ("y_train ===\n\n"), y_train
print ("x_test ===\n\n"), X_test
print ("y_test ===\n\n"), y_test

print ("x_train size ===\n\n"), len(X_train)
print ("y_train size ===\n\n"), len(y_train)
print ("X_test size ===\n\n"), len(X_test)
print ("y_test size ===\n\n"), len(y_test)



in_out_neurons = 3  
hidden_neurons = 300

model = Sequential()  
model.add(LSTM(in_out_neurons, hidden_neurons, return_sequences=False))  
model.add(Dense(hidden_neurons, in_out_neurons))  
model.add(Activation("linear"))  
model.compile(loss="mean_squared_error", optimizer="rmsprop")  





# and now train the model
# batch_size should be appropriate to your memory size
# number of epochs should be higher for real world problems
model.fit(X_train, y_train, batch_size=100, nb_epoch=20, validation_split=0.05) 


predicted = model.predict(X_test)  
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))

# and maybe plot it
plt.plot(pd.DataFrame(predicted[:20])) 
plt.plot(pd.DataFrame(y_test[:20]))
plt.show()


