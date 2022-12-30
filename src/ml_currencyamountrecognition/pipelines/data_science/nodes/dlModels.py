from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Dropout
from tensorflow.keras.layers import Reshape


'''
Constructs a CNN model based on 5 layers.
Args : 
    - X_train : features of the training set
    - X_val : features of the validation set
    - y_train : labels of the training set
    - y_val : labels of the validation set
Returns : 
    dataframe : the model
'''
def construct_CNN_model1(X_train, X_val, y_train, y_val):
    print("********** X TRAIN *************** \n", X_train, "\n ************** TYPE ************** ", type(X_train))
    model = Sequential()
    model.add(Reshape((X_train.shape[1], X_train.shape[2], 1), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    return model


'''
Constructs a CNN model based on 6 layers.
Args : 
    - X_train : features of the training set
    - X_val : features of the validation set
    - y_train : labels of the training set
    - y_val : labels of the validation set
Returns : 
    dataframe : the model
'''
def construct_CNN_model2(X_train, X_val, y_train, y_val):
    model = Sequential()
    model.add(Reshape((X_train.shape[1], X_train.shape[2], 1), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Flatten())
    model.add(Dense(2, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    return model



'''
Constructs a LSTM model based on 2 layers.
Args : 
    - X_train : features of the training set
    - X_val : features of the validation set
    - y_train : labels of the training set
    - y_val : labels of the validation set
Returns : 
    dataframe : the model
'''
def construct_LSTM_model1(X_train, X_val, y_train, y_val):
    model = Sequential()
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    return model


'''
Constructs a LSTM model based on 3 layers.
Args : 
    - X_train : features of the training set
    - X_val : features of the validation set
    - y_train : labels of the training set
    - y_val : labels of the validation set
Returns : 
    dataframe : the model
'''
def construct_LSTM_model2(X_train, X_val, y_train, y_val):
    model = Sequential()
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    return model


'''
Constructs a feedward neural network model based on 9 layers.
Args : 
    - X_train : features of the training set
    - X_val : features of the validation set
    - y_train : labels of the training set
    - y_val : labels of the validation set
Returns : 
    dataframe : the model
'''
def construct_feedward_neural_network_model(X_train, X_val, y_train, y_val):
    model = Sequential()
    model.add(Dense(64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Flatten())
    model.add(Dense(2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Dense(2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    return model