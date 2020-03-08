from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense,Convolution2D,Activation,MaxPooling2D,GlobalAveragePooling2D

def get_CNN_Model():
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, border_mode="same", init='he_normal', input_shape=(300, 300, 3), dim_ordering="tf"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))
    model.add(Convolution2D(36, 5, 5))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))
    model.add(Convolution2D(48, 5, 5))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation("relu"))
    model.add(GlobalAveragePooling2D());
    model.add(Dense(500, activation="relu"))
    model.add(Dense(90, activation="relu"))
    model.add(Dense(30,activation="sigmoid"))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss ='mse', metrics = ['accuracy'])
    return model
