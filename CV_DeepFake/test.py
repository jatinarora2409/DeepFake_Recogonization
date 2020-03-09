from face_detect_util.get_face import get_frames, get_faces,get_cropped_images
from CNN.model import get_CNN_Model, get_CNN_Model_ForClassification,getRNNModel
import numpy as np
import sys
from keras.models import load_model,Model

framesFromFile1 = 300
framesFromFile2 = 300

height = 200
width = 200

file1 = '../../manipulated_sequences/Deepfakes/raw/videos/183_253.mp4'
file2 = '../../original_sequences/youtube/raw/videos/183.mp4'


def train_model():
    np.set_printoptions(threshold=sys.maxsize)
    frames = get_frames(file1, framesFromFile1, startingPoint=0)
    facesIncorrect = get_faces(frames,height=height,width=width)
    facesIncorrect = np.asarray(facesIncorrect)
    del frames
    
    frames = get_frames(file2, framesFromFile2)
    facesCorrect = get_faces(frames, height=height, width=width)
    facesCorrect = np.asarray(facesCorrect)
    del frames


    count_incorrect = len(facesIncorrect)
    count_correct = len(facesCorrect)




    # print(labels)
    # print(len(faces))
    # print(len(facesCorrect))



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
    epochs = 100;
    print(y_train)
    model.fit(x_train[s], y_train[s], validation_split=0.2, shuffle=True, epochs=epochs, batch_size=20, verbose=1)
    model.save('classification.h5')


def test_model():
    model = load_model('classification.h5')
    correct_test_faces = 10
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

train_model()
test_model()


