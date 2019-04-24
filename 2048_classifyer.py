import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle

from keras import models, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint

# import Reccurent_Network

'''
need to make one hot encoding universal
#convolutional model accuracy - 68.0ish %

'''

#getting data 
df = pd.read_csv('2048_Training_Data_2.csv',index_col=[0])
df_2 = pd.read_csv('2048_Training_Data_3.csv',index_col=[0])
df_3 = pd.read_csv('2048_Training_Data_4.csv',index_col=[0])


df = pd.DataFrame(df[:200])
print df.shape
print df_2.shape

# df = pd.concat([df,df_2,df_3])

print df.shape

def pre_processing(df):
	#dropping score column
	df = df.drop(columns = 'score')
	# print df.head(20)


	#getting features
	X = df.drop(columns = 'lastMove')
	# print X.head(5)

	X = X / 2048.0

	# categories = ['w','a','d','s']

	#one hot encoding target
	enc = OneHotEncoder()


	print df['lastMove'].head()
	enc.fit(df[['lastMove']])

	filename = 'one_hot.sav'
	pickle.dump(enc, open(filename, 'wb'))

	y = enc.transform(df[['lastMove']]).toarray()

	


import itertools
# np.array(list(itertools.zip_longest(*v, fillvalue=0))).T
def Reccurent_Network(df):
	training_length = 30
	num_keys = 4

	x_list = []
	y_list = []


	#one hot encoding
	dummy = pd.get_dummies(df['lastMove'])
	df = df.merge(dummy,left_index = True, right_index = True)

	#dropping unnessary columns
	df = df.drop(columns = ['score','lastMove'])
	

	#creating targets and features in a series form
	for i in ( np.arange(len(df) - training_length) ):
		x_list.append(df.iloc[i:(i + training_length),0:16])
		y_list.append(df.iloc[i:(i + training_length),16:20])

	x_list_2 = []
	y_list_2 = []
	
	for i in x_list:
		y = i.values
		x_list_2.append(y)
	x_list_2 = np.array(x_list_2)

	print type(x_list_2)
	# print x_list_2.shape()

	for i in y_list:
		y = i.values
		y_list_2.append(y)
	y_list_2 = np.array(y_list_2)



	model = Sequential()
	print np.shape(x_list_2)
	print np.shape(y_list_2)

	# x_list_2 = x_list_2.reshape(170,30,16)


	# # Embedding layer
	model.add(
	    Embedding(input_dim=(30,),
              input_length = training_length,
              output_dim=(3),
              
              trainable=False,
              mask_zero=True))

	# # Masking layer for pre-trained embeddings
	model.add(Masking(mask_value=0.0))

	# Recurrent layer KERAS RL
	model.add(LSTM( 170,input_dim=(30,16) ))

	# model.add(LSTM(10, input_shape=(x_list_2.shape[1:] )))
	               

	# Fully connected layer
	model.add(Dense(30, activation='relu'))

	# Dropout for regularization
	model.add(Dropout(0.5))

	model.add(Flatten())

	# Output layer
	model.add(Dense(num_keys, activation='softmax'))

	model.compile(
	    optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	X_train, X_test, y_train, y_test = train_test_split(np.array(x_list_2), np.array(y_list_2), test_size=.25, random_state=42)

	# print X_train[0:4]

	# X_train = np.array(X_train)
	# y_train = np.array(y_train)

	# # Create callbacks
	# callbacks = [EarlyStopping(monitor='val_loss', patience=5),
 #             ModelCheckpoint('../models/model.h5'), save_best_only = True, save_weights_only=False]

	history = model.fit(X_train,  y_train, 
                    batch_size=100, epochs=150,
                    validation_data=(X_test, y_test) )


 	print model.evaluate(X_test, y_test)

	print type(sequenceList)




Reccurent_Network(df)



#splitting data into sequences


def convolutional_net(X,y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
	network = models.Sequential()
	network.add(layers.Dense(160,activation='relu',input_shape=(16,)))
	# network.add(layers.Dense(16,activation='tanh',input_shape=(16,)))
	network.add(layers.Dense(4,activation='softmax'))

	network.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

	# help(network.compile)
	    

	# If we reload this right before fitting the model, the model will start from scratch
	# network.save_weights('model_init.h5')
	    
	#Early stopping
	patienceCount = 10
	   
	callbacks = [EarlyStopping(monitor='val_loss', patience=patienceCount),
	                     ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', 
	                     save_best_only=True)]

	# train_labels_cat = to_categorical(train_labels)
	# test_labels_cat = to_categorical(test_labels)

	history = network.fit(X_train,y_train,
	                            epochs=1500,
	                            batch_size=128,
	                            validation_data=(X_test,y_test))

	acc = history.history['acc'][-patienceCount + 1]

	network.save_weights('Conv_Net_1.h5')
	
	print('Accuracy: ', acc)


def random_forest(X,y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

	clf = RandomForestClassifier(n_estimators=300, max_depth=4,
                            random_state=0)

	clf.fit(X_train, y_train)

	print clf.score(X_test,y_test)

	#splitting data
	filename = 'random_forest_model_simple.sav'
	pickle.dump(clf, open(filename, 'wb'))


def Reccurent_Network(X,y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

	model = Sequential()
	model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
	model.add(LSTM(hidden_size, return_sequences=True))
	model.add(LSTM(hidden_size, return_sequences=True))
	if use_dropout:
	    model.add(Dropout(0.5))
	model.add(TimeDistributed(Dense(vocabulary)))
	model.add(Activation('softmax'))

# random_forest(X,y)
# convolutional_net(X,y)







