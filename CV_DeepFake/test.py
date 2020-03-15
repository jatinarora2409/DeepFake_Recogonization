from face_detect_util.get_face import get_frames, get_faces,get_cropped_images
from CNN.model import get_CNN_Model, get_CNN_Model_ForClassification,getRNNModel
import numpy as np
import sys
from keras.models import load_model,Model
import os
from os.path import isfile, join

framesFromFile1 = 500
framesFromFile2 = 500

height = 200
width = 200

def get_all_files(folder):
    filepaths = [os.path.join(folder, f) for f in os.listdir(folder)]
    return filepaths

def train_model(files_original,files_fake):
    np.set_printoptions(threshold=sys.maxsize)
    tempFaces = []
    print( files_original)
    print(files_fake)
    for original_file in files_original:
        frames = get_frames(original_file, framesFromFile1, startingPoint=0)
        tempFaces.extend(get_faces(frames,height=height,width=width))
        del frames

    facesCorrect = np.asarray(tempFaces);
    tempFaces = [];
    for fake_file in files_fake:
        frames = get_frames(fake_file, framesFromFile2)
        tempFaces.extend(get_faces(frames, height=height, width=width))
        del frames
    facesIncorrect = np.asarray(tempFaces)

    count_incorrect = len(facesIncorrect)
    count_correct = len(facesCorrect)
    print("count_incorrect")
    print(count_incorrect)
    print("\n\n count_correct")
    print(count_correct)

    x_train = np.concatenate((facesIncorrect, facesCorrect))
    labels = []

    for i in range (0,(count_incorrect)):
        labels.append([0,1])

    for i in range(0, (count_correct)):
        labels.append([1,0])

    y_train = np.asarray(labels)

    s = np.arange(len(x_train));
    np.random.shuffle(s)

    model = getRNNModel(height,width,3)
    epochs = 50;
    print(y_train)
    model.fit(x_train[s], y_train[s], validation_split=0.2, shuffle=True, epochs=epochs, batch_size=20, verbose=1)
    model.save('classification.h5')


def test_model():
    model = load_model('classification.h5')
    correct_test_faces = 10
    file1 = '../manipulated_sequences/Deepfakes/raw/videos/469_481.mp4'
    file2 = '../original_sequences/youtube/raw/videos/481.mp4'
    framesTest = get_frames(file1, correct_test_faces, startingPoint=100)
    facesCorrectTest = get_cropped_images(framesTest, height=height, width=width)
    facesCorrectTest = np.asarray(facesCorrectTest)

    incorrect_test_faces = 10
    framesTest = get_frames(file2, incorrect_test_faces, startingPoint=100)
    facesInCorrectTest = get_faces(framesTest, height=height, width=width)
    facesInCorrectTest = np.asarray(facesInCorrectTest)
    X_test = np.concatenate((facesCorrectTest, facesInCorrectTest))
    del framesTest

    y_test_result = model.predict(X_test)
    print("result:", y_test_result)

files_fake = get_all_files('../manipulated_sequences/Deepfakes/raw/videos/')
files_original = get_all_files('../original_sequences/youtube/raw/videos/')

train_model(files_original,files_fake)
test_model()
