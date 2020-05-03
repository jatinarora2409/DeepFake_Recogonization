from face_detect_util.get_face import get_frames, get_faces,get_cropped_images
from CNN.model import get_CNN_Model, get_CNN_Model_ForClassification,getRNNModel,getCNNInceptionModel,getLSTMModel,getCustomCNNLSTMModel
import numpy as np
import sys
from keras.models import load_model,Model
import os
import matplotlib.pyplot as plt

import cv2
from os.path import isfile, join

framesFromFile1 = 200
framesFromFile2 = 200

height = 320
width = 320

number_of_faces = 40

def get_all_files(folder):
    filepaths = [os.path.join(folder, f) for f in os.listdir(folder)]
    return filepaths

def get_faces_local(files_original,files_fake):
    np.set_printoptions(threshold=sys.maxsize)
    tempFaces = []
    print(files_original)
    print(files_fake)

    labels = []

    for original_file in files_original:
        frames = get_frames(original_file, startingPoint=0)
        faces = get_faces(frames, height=height, width=width,number_of_faces=number_of_faces)
        if(len(faces)==number_of_faces):
           tempFaces.extend(faces)
           labels.append([0, 1])
        del frames
        del faces

    facesCorrect = np.asarray(tempFaces);
    tempFaces = [];
    for fake_file in files_fake:
        frames = get_frames(fake_file,startingPoint=0)
        faces = get_faces(frames, height=height, width=width, number_of_faces=number_of_faces)
        if (len(faces) == number_of_faces):
            tempFaces.extend(faces)
            labels.append([1, 0])
        del frames
        del faces

    facesIncorrect = np.asarray(tempFaces)
    count_incorrect = len(facesIncorrect)
    count_correct = len(facesCorrect)
    print(facesIncorrect.shape)
    x_train = np.concatenate((facesIncorrect, facesCorrect))
    return x_train,count_incorrect,count_correct,labels


def get_faces_local_for_CNN(files_original,files_fake):
    np.set_printoptions(threshold=sys.maxsize)
    tempFaces = []

    labels = []

    for original_file in files_original:
        frames = get_frames(original_file, startingPoint=0)
        faces = get_faces(frames, height=height, width=width,number_of_faces=number_of_faces)
        if(len(faces)==number_of_faces):
           tempFaces.extend(faces)
           for face in faces:
               labels.append([0, 1])

        del frames
        del faces

    facesCorrect = np.asarray(tempFaces);
    tempFaces = [];
    for fake_file in files_fake:
        frames = get_frames(fake_file,startingPoint=0)
        faces = get_faces(frames, height=height, width=width, number_of_faces=number_of_faces)
        if (len(faces) == number_of_faces):
            tempFaces.extend(faces)
            for face in faces:
                labels.append([1, 0])
        del frames
        del faces

    facesIncorrect = np.asarray(tempFaces)
    count_incorrect = len(facesIncorrect)
    count_correct = len(facesCorrect)
    print(facesIncorrect.shape)
    x_train = np.concatenate((facesCorrect, facesIncorrect))
    return x_train,count_incorrect,count_correct,labels


def train_model_CNN_LSTM(files_original,files_fake):
    original_file_iter = iter(files_original)
    files_fake_iter = iter(files_fake)
    original_file = next(original_file_iter,None)
    fake_file = next(files_fake_iter,None)
    count = 0;
    original_file_array = []
    fake_file_array=[]
    CNN_LSTM_model = getCustomCNNLSTMModel(number_of_faces,height,width,3)
    while original_file is not None or fake_file is not None :
        if original_file is not None:
            original_file_array.append(original_file)
        if fake_file is not None:
            fake_file_array.append(fake_file)
        count = count + 1
        original_file = next(original_file_iter, None)
        fake_file = next(files_fake_iter, None)
        if(count==1):
            x_train,count_incorrect,count_correct,labels = get_faces_local(original_file_array,fake_file_array)
            x_train = np.array(x_train)
            print("\n\n")
            print("Shape of X_train: " + str(x_train.shape))
            print("Shape of X_train, single frame " + str(x_train[0].shape))

            #input_for_LSTM = CNN_model.predict(x_train);
            #print("Shape of input_for_LSTM before reshape"+str(input_for_LSTM.shape))
            input_for_LSTM = x_train.reshape(len(labels),number_of_faces,height,width,3)

           #print("Shape of input_for_LSTM after reshape"+str(input_for_LSTM.shape))
            y_train = np.asarray(labels)
            print("Training on labels")
            print(labels)
            print("\n\n")
            CNN_LSTM_model.fit(input_for_LSTM, y_train,validation_split=0.2, shuffle=True, epochs=30, verbose=1)
            count = 0
            original_file_array = []
            fake_file_array = []
            del x_train
            #del input_for_LSTM
            del labels

    CNN_LSTM_model.save('CNN_lstmModel.h5')


def test_model_CNN_Lstm(files):
    model = load_model('CNN_lstmModel.h5')
    count = 0
    tempFaces = []
    for file in files:
        print("File")
        frames = get_frames(file, number_of_frames=-1, startingPoint=0)
        faces = get_faces(frames, height=height, width=width,number_of_faces=number_of_faces)
        if(len(faces)!=number_of_faces):
            print("File: " + str(file))
            print("No Result Found\n")
            continue;
        else:
            tempFaces.extend(faces)
            count = count+1;

    testFaces = np.asarray(tempFaces)
    input_for_LSTM = testFaces.reshape(count, number_of_faces, height, width, 3)
    print("Shape for testing:", input_for_LSTM.shape)
    y_test_result = model.predict(input_for_LSTM)
    print("\n\n")
    print("Result: ")
    print(y_test_result)
    print("\n")



def train_model_RNN_or_CNN(files_original,files_fake):
    original_file_iter = iter(files_original)
    files_fake_iter = iter(files_fake)
    original_file = next(original_file_iter, None)
    fake_file = next(files_fake_iter, None)
    model = get_CNN_Model(height, width,1)
    count = 0;
    original_file_array = []
    fake_file_array = []
    # CNN_model = getCNNInceptionModel(height, width, 3)
    # LSTM_model = getLSTMModel();
    # CNN_LSTM_model = getCustomCNNLSTMModel(number_of_faces, height, width, 3)
    while original_file is not None or fake_file is not None:
        if original_file is not None:
            original_file_array.append(original_file)
        if fake_file is not None:
            fake_file_array.append(fake_file)
        count = count + 1
        original_file = next(original_file_iter, None)
        fake_file = next(files_fake_iter, None)
        print("Count: "+str(count))
        if (count == 1):
            print("Took 3 Out")
            x_train,count_incorrect,count_correct,labels = get_faces_local_for_CNN(original_file_array,fake_file_array)
            y_train = np.asarray(labels)
            x_train_shape = x_train.shape
            #x_train = x_train.reshape(x_train_shape[0], x_train_shape[1], x_train_shape[2], 1)
            show_input(x_train, y_train)
            epochs = 40;
            model.fit(x_train, y_train, validation_split=0.2, shuffle=True, epochs=epochs, batch_size=20, verbose=1)
            count = 0
            original_file_array = []
            fake_file_array = []
            del x_train
    # del input_for_LSTM
            del labels

    model.save('classification_CNN.h5')


def show_input(x_train,y_train):
    for i in range(0,len(x_train)):
        # if(y_train[i][0]==1):
            fig, ax = plt.subplots(1, 1, figsize=(15, 15))
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            plt.grid(False)
            ax.imshow(x_train[i])
            print(y_train[i])
            plt.show()

def test_model(files):
    model = load_model('classification_CNN.h5')
    for file in files:
        frames = get_frames(file, number_of_frames=40, startingPoint=0)
        tempFaces = (get_faces(frames,height=height,width=width))
        testFaces = np.asarray(tempFaces)
        testFaces_shape = testFaces.shape
        testFaces = testFaces.reshape(testFaces_shape[0], testFaces_shape[1], testFaces_shape[2], 1)
        print(testFaces[0].shape)
        #testFaces = np.concatenate(testFaces)
        print(testFaces.shape)
        y_test_result = model.predict(testFaces)
        count_fake = 0
        count_positive = 0
        count_clear_fake = 0
        count_clear_positive = 0

        for frame_result in y_test_result:

            if (frame_result[0]-frame_result[1]>=0.3):
               count_clear_positive = count_clear_positive+1
            elif (frame_result[0]-frame_result[1]>0):
                count_positive = count_positive+1
            elif(frame_result[1]-frame_result[0]>=0.3):
                count_clear_fake = count_clear_fake+1
            elif(frame_result[1]-frame_result[0]>0):
                count_fake = count_fake+1
            else:
                count_doubt = count_doubt+1

        print("File: "+file)
        print("count_clear positive:" + str(count_clear_positive))
        print("count_positive: " +str(count_positive))
        print("count_fake: " + str(count_fake))
        print("count_clear_fake: " + str(count_clear_fake))

        print("\n")
        del frames


    # correct_test_faces = 10
    # file1 = '../manipulated_sequences/Deepfakes/raw/videos/469_481.mp4'
    # file2 = '../original_sequences/youtube/raw/videos/481.mp4'
    # framesTest = get_frames(file1, correct_test_faces, startingPoint=100)
    # facesCorrectTest = get_cropped_images(framesTest, height=height, width=width)
    # facesCorrectTest = np.asarray(facesCorrectTest)
    #
    # incorrect_test_faces = 10
    # framesTest = get_frames(file2, incorrect_test_faces, startingPoint=100)
    # facesInCorrectTest = get_faces(framesTest, height=height, width=width)
    # facesInCorrectTest = np.asarray(facesInCorrectTest)
    # X_test = np.concatenate((facesCorrectTest, facesInCorrectTest))
    # del framesTest
    #
    # y_test_result = model.predict(X_test)
    # print("result:", y_test_result)



def train_model_CNN_LSTM_New(files_original,files_fake):
    files = []
    files.extend(files_original)
    files.extend(files_fake)
    np.random.shuffle(files)
    CNN_LSTM_model = getCustomCNNLSTMModel(number_of_faces,height,width,3)
    for file in files:
        frames = get_frames(file, startingPoint=0)
        faces = get_faces(frames, height=height, width=width, number_of_faces=number_of_faces)
        faces = np.asarray(faces);
        faces = np.array(faces);
        if (len(faces) != number_of_faces):
            continue
        if(file  in files_original):
            label  = [0, 1]
        if(file in files_fake):
            label = [1, 0]
        input_for_LSTM = faces.reshape(1, number_of_faces, height, width, 3)
        labels = []
        labels.append(label)
        y_train = np.asarray(labels)
        CNN_LSTM_model.fit(input_for_LSTM, y_train, epochs=20, verbose=1)
    CNN_LSTM_model.save('CNN_New_lstmModel.h5')



files_fake = get_all_files('../manipulated_sequences/Deepfakes/raw/videos/')
files_original = get_all_files('../original_sequences/youtube/raw/videos/')
file_original = ['../original.mp4']
file_fake = ['../deepfake.mp4']
train_model_CNN_LSTM_New(files_original,files_fake)
test_files = get_all_files('../test_files/')
test_model(test_files)

def check_output(file):
    img = cv2.imread(file)
    model = getCNNInceptionModel(img.shape[0],img.shape[1],img.shape[2])
    input_predict = np.asarray([img]);
    print("\n\n Input Predict Shape")
    print(input_predict.shape);
    output = model.predict(input_predict);
    print("\n\n SHAPE: ");
    print(output.shape);
    print("\n\n OUTPUT: ");
    print(output);

#check_output('./car.jpg')
