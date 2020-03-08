from face_detect_util.get_face import get_frames,get_faces
from CNN.model import get_CNN_Model
import numpy as np

incorrectFaces=5
correctFaces=5

frames = get_frames('/Users/jatinarora/CV_DeepFake/manipulated_sequences/Deepfakes/raw/videos/183_253.mp4',incorrectFaces)
faces = get_faces(frames,height=300,width=300)
del frames

frames = get_frames('/Users/jatinarora/CV_DeepFake/original_sequences/youtube/raw/videos/183.mp4',correctFaces)
facesCorrect = get_faces(frames,height=300,width=300)
del frames

faces.append(facesCorrect)

del facesCorrect

labels = np.ones((incorrectFaces+correctFaces,), dtype=int)
labels[0:incorrectFaces] = 0
#print(labels)

#shuffling the data
s = np.arange(faces.shape[0]);
np.random.shuffle(s)


model = get_CNN_Model()
epochs = 30;
hist = model.fit(faces[s], labels[s], validation_split=0.2, shuffle=True, epochs=epochs, batch_size=20, verbose=1)