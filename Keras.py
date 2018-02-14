from keras.models import Sequential
import numpy as np
from keras.utils import np_utils
from keras import callbacks
import tensorflow as tf
from keras.layers.core import Dense, Activation

#Visualization
remote = callbacks.RemoteMonitor(root='http://localhost:9000')

#Initialize tf with python_io
tf.python_io = tf

#Set a random seed
np.random.seed(47)

#Initialize the Sequential model
xor = Sequential()

#Initialize our data
X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y = np.array([[0],[1],[1],[0]]).astype('float32')

#Add the required layers
xor.add(Dense(8, input_dim=2))
xor.add(Activation(activation='tanh'))
xor.add(Dense(1))
xor.add(Activation(activation='sigmoid'))

#Compile the model
xor.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
xor.summary()

#Fitting the model
history = xor.fit(X,y, epochs=100, verbose=1, callbacks=[remote])

#Scoring the model
score = xor.evaluate(X,y)
print("\nThe accuracy socre is: ",score[-1])

#Checking the predictions
print("\n Prediction for X is: ", xor.predict_proba(X))
