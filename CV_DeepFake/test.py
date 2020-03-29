from face_detect_util.get_face import get_frames, get_faces,get_cropped_images
from CNN.model import get_CNN_Model, get_CNN_Model_ForClassification,getRNNModel,getCNNInceptionModel,getLSTMModel
import numpy as np
import sys
from keras.models import load_model,Model
import os
import cv2
from os.path import isfile, join

framesFromFile1 = 500
framesFromFile2 = 500

height = 299
width = 299

def get_all_files(folder):
    filepaths = [os.path.join(folder, f) for f in os.listdir(folder)]
    return filepaths

def get_faces_local(files_original,files_fake):
    np.set_printoptions(threshold=sys.maxsize)
    tempFaces = []
    print(files_original)
    print(files_fake)
    for original_file in files_original:
        frames = get_frames(original_file, startingPoint=0,number_of_frames=40)
        tempFaces.extend(get_faces(frames, height=height, width=width))
        print("TempFaces Size: " +str(len(tempFaces)))
        del frames

    facesCorrect = np.asarray(tempFaces);
    tempFaces = [];
    for fake_file in files_fake:
        frames = get_frames(fake_file, number_of_frames=40,startingPoint=0)
        tempFaces.extend(get_faces(frames, height=height, width=width))
        print("TempFaces Size: " + str(len(tempFaces)))
        del frames
    facesIncorrect = np.asarray(tempFaces)

    count_incorrect = len(facesIncorrect)
    count_correct = len(facesCorrect)
    print("count_incorrect")
    print(count_incorrect)
    print("\n\n count_correct")
    print(count_correct)
    x_train = np.concatenate((facesIncorrect, facesCorrect))
    return x_train,count_incorrect,count_correct


def train_model_CNN_LSTM(files_original,files_fake):
    x_train,count_incorrect,count_correct = get_faces_local(files_original,files_fake)
    labels = []
    for i in range (0,len(files_fake)):
        labels.append([0,1])

    for i in range(0,len (files_original)):
        labels.append([1,0])

    CNN_model = getCNNInceptionModel(height,width,3)
    input_for_LSTM = CNN_model.predict(x_train);
    input_for_LSTM = input_for_LSTM.reshape(len(files_original) + len(files_fake),40,2048)
    y_train = np.asarray(labels)


    LSTM_model = getLSTMModel();
    LSTM_model.fit(input_for_LSTM, y_train,validation_split=0.2, shuffle=True, epochs=30, verbose=1)
    LSTM_model.save('lstmModel.h5')


def test_model_CNN_RNN(files):
    model = load_model('lstmModel.h5')
    CNN_model = getCNNInceptionModel(height, width, 3)
    for file in files:
        tempFaces = []
        frames = get_frames(file, number_of_frames=40, startingPoint=0)
        tempFaces.extend(get_faces(frames, height=height, width=width))
        testFaces = np.asarray(tempFaces)
        input_for_LSTM = CNN_model.predict(testFaces);
        input_for_LSTM = input_for_LSTM.reshape(1, 40, 2048)
        y_test_result = model.predict(input_for_LSTM)
        print("\n\n")
        print("File: "+str(file))
        print("Result: ")
        print(y_test_result)
        print("\n")



def train_model_RNN(files_original,files_fake):
    x_train,count_incorrect,count_correct = get_faces_local(files_original,files_fake)
    labels = []

    for i in range (0,(count_incorrect)):
        labels.append([0,1])

    for i in range(0, (count_correct)):
        labels.append([1,0])

    y_train = np.asarray(labels)

    s = np.arange(len(x_train));
    np.random.shuffle(s)

    model = getRNNModel(height,width,3)
    epochs = 20;
    print(y_train)
    model.fit(x_train[s], y_train[s], validation_split=0.2, shuffle=True, epochs=epochs, batch_size=20, verbose=1)
    model.save('classification.h5')


def test_model(files):
    model = load_model('classification.h5')
    for file in files:
        tempFaces = []
        frames = get_frames(file, number_of_frames=-1, startingPoint=0)
        tempFaces.extend(get_faces(frames,height=height,width=width))
        testFaces = np.asarray(tempFaces)
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

files_fake = get_all_files('../manipulated_sequences/Deepfakes/raw/videos/')
files_original = get_all_files('../original_sequences/youtube/raw/videos/')
file_original = ['../original.mp4']
file_fake = ['../deepfake.mp4']
#train_model_CNN_LSTM(files_original,files_fake)

test_files = get_all_files('../test_files/')
test_model_CNN_RNN(test_files)

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
