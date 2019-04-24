

from keras.layers import Conv1D, MaxPooling1D,GlobalAveragePooling1D

model.add(
    Embedding(input_dim=(30,16),
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
    



























