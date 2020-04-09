from keras.callbacks import ModelCheckpoint
from keras.models import Sequential,Model
from keras.layers import Dense,Convolution2D,Activation,MaxPooling2D,GlobalAveragePooling2D,Conv2D,Flatten,Dropout,LSTM
from keras import applications
from keras.optimizers import SGD, Adam


def get_CNN_Model(height,width):
    model = Sequential()
    model.add(Convolution2D(24, (5,5), border_mode="same", init='he_normal',
                            input_shape=(height,width, 3),data_format="channels_last"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))
    model.add(Convolution2D(36, (5, 5)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))
    model.add(Convolution2D(48, (5, 5)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(GlobalAveragePooling2D());
    model.add(Dense(500, activation="relu"))
    model.add(Dense(90, activation="relu"))
    model.add(Dense(30,activation="sigmoid"))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss ='mse', metrics = ['accuracy'])
    return model


def get_CNN_Model_ForClassification(height,width):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape = (height, width, 3)))
    model.add(Conv2D(32, kernel_size=3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def getRNNModel(height,width,channels):
    base_model = applications.resnet50.ResNet50(weights=None, include_top=False, input_shape=(height, width, channels))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model




def getCNNInceptionModel(height,width,channels):
    base_model=applications.inception_v3.InceptionV3(weights = "imagenet",include_top = False,input_shape=(height, width, channels),pooling='max')
    return base_model

def getLSTMModel():
    model = Sequential()
    model.add(LSTM(16, input_shape=(80,2048), dropout=0.5,))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    x = model.output
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)
    adam = Adam(lr=0.00001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model