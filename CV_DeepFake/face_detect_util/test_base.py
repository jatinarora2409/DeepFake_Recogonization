from IPython.core.display import display
from sklearn.metrics import log_loss
import cv2 as cv
import os
import matplotlib.pylab as plt
import face_recognition
from PIL import Image,ImageDraw

def get_all_files(folder):
    filepaths = [os.path.join(folder, f) for f in os.listdir(folder)]
    return filepaths


files_fake = get_all_files('../../manipulated_sequences/Deepfakes/raw/videos/')
files_original = get_all_files('../../original_sequences/youtube/raw/videos/')



plt.style.use('ggplot')
fig, ax = plt.subplots(1,1, figsize=(15, 15))
# video_file = train_video_files[30]

## Getting the first frame
cap = cv.VideoCapture(files_original[0])
success, image = cap.read()
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
cap.release()
ax.imshow(image)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
plt.grid(False)
plt.show()
face_locations = face_recognition.face_locations(image)
print("I found {} face(s) in this photograph.".format(len(face_locations)))

for face_location in face_locations:

    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    fig, ax = plt.subplots(1,1, figsize=(5, 5))
    plt.grid(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(face_image)
    plt.show()

face_landmarks_list = face_recognition.face_landmarks(face_image)
pil_image = Image.fromarray(face_image)
d = ImageDraw.Draw(face_image)
for face_landmarks in face_landmarks_list:
    # Print the location of each facial feature in this image
    for facial_feature in face_landmarks.keys():
        if(facial_feature=="chin"):
          print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

    # Let's trace out each facial feature in the image with a line!
    for facial_feature in face_landmarks.keys():
        if (facial_feature == "chin"):
            d.line(face_landmarks[facial_feature], width=3)

# Show the picture
pil_image.show()