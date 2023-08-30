from idlelib import history

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold
import keras
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


df = pd.read_csv("new_dataset.csv")
df = df.sample(frac=1)

norm_dataset = StandardScaler().fit_transform(X=df)
#scaler = MinMaxScaler()
#norm_dataset = scaler.fit_transform(df)

X = norm_dataset[:, :-1]
Y = df.iloc[:, 17:18].values
onehotencoder = OneHotEncoder()
Y = onehotencoder.fit_transform(Y).toarray()



kfold = KFold(n_splits=5, shuffle=True)

rmseList = []
rrseList = []
Temp = []
val_Temp = []

for i, (train, test) in enumerate(kfold.split(X)):


    model = Sequential()

    model.add(Dense(22, activation="relu", input_shape=(17, )))
    model.add(Dense(5, activation="softmax", input_dim=22))

    keras.optimizers.legacy.SGD(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    callback = keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)

    history = model.fit(X[train], Y[train], validation_split=0.1, epochs=100, batch_size=500, shuffle=True, verbose=2, callbacks=[callback])


    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model mse')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['mse', 'val_mse'], loc='upper left')
    plt.show()

    # Evaluate model
    scores = model.evaluate(X[test], Y[test], verbose=2)

    #print(scores)
    rmseList.append(scores[1])
    print("Fold :", i, " Accuracy:", scores[1])


print("Accuracy: ", np.mean(rmseList))






