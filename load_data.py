# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:56:08 2020

@author: IniLaptop
"""
import json
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import time

DATA_PATH = "./Extracted_data.json"

def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y

def prepare_datasets(test_size, validation_size):

    # load data
    X, y = load_data(DATA_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):

    # build network topology
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(128))

    # dense layer
    model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dropout(0.3))
    
    #

    # output layer
    model.add(keras.layers.Dense(5, activation='softmax'))

    return model


if __name__ == "__main__":
    begin = time.time() 
    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.01, 0.3)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2]) # 130, 13
    model = build_model(input_shape)

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.001 ,amsgrad=True)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    model.fit(X_train, 
              y_train, 
              validation_data=(X_validation, y_validation), 
              batch_size=32, 
              epochs=50)
 
    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_loss)
    
    #Datatest MFCC features
    # print('\nData test   :', X_test)
        
    #Predicting output from test sets
    pred = model.predict(X_test) 
    y_class = [0,1,2,3,4]
    Y_Class = [0]*len(X_test)
    
    # Real data test
    print('\nData test   :', y_test)
    # print('\nData Predict:', pred)
    
    for i in range(len(X_test)):
        
        subarr_X_test =pred[i]
        temp = y_class[np.argmax(subarr_X_test)]
        Y_Class[i] = temp
        
    # Predicted Label
    print('\nlabel Predict:'," ".join(map(str,Y_Class)))
    
    def find_class_accuracy(label_index, actual, prediction):
        label_count = 0
        correct_count = 0
        for i in range(len(actual)):
            if actual[i] == label_index:
                label_count += 1
                if actual[i] == prediction[i]:
                    correct_count += 1
        result = correct_count / label_count
        return result, correct_count, label_count
    
    for i in range(len(y_class)):
        result, correct_count, label_count = find_class_accuracy(i, y_test, Y_Class)
        print("Akurasi dari kelas {}: {}".format(y_class[i], result * 100))
    
    time.sleep(1) 
    # store end time 
    end = time.time() 
      
    # total time taken 
    print(f"\nTotal runtime of the program is {end - begin} sec")
