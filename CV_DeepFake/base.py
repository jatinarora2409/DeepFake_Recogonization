from IPython.core.display import display
from sklearn.metrics import log_loss
import cv2 as cv
import os
import matplotlib.pylab as plt
import face_recognition
from PIL import Image,ImageDraw
from face_detect_util.get_face import get_frames, get_faces,get_cropped_images

height = 299
width=299
number_of_faces=40

def get_all_files(folder):
    filepaths = [os.path.join(folder, f) for f in os.listdir(folder)]
    return filepaths

files_fake = get_all_files('../manipulated_sequences/Deepfakes/raw/videos/')
files_original = get_all_files('../original_sequences/youtube/raw/videos/')

for file_fake,file_original in zip(files_fake,files_original):
    frames = get_frames(file_fake, startingPoint=0)
    faces = get_faces(frames, height=height, width=width, number_of_faces=number_of_faces)
    if(len(faces)==number_of_faces):
        for face in faces :
            fig, ax = plt.subplots(1, 1, figsize=(15, 15))
            ax.imshow(face)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.title.set_text(f"FRAME 0: {file_fake.split('/')[-1]}")
            plt.grid(False)
            plt.show()

    frames = get_frames(file_original, startingPoint=0)
    faces = get_faces(frames, height=height, width=width, number_of_faces=number_of_faces)
    if (len(faces) == number_of_faces):
        for face in faces:
            fig, ax = plt.subplots(1, 1, figsize=(15, 15))
            ax.imshow(face)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.title.set_text(f"FRAME 0: {files_original.split('/')[-1]}")
            plt.grid(False)
            plt.show()

video_file = '/Users/jatinarora/CV_DeepFake/manipulated_sequences/Deepfakes/raw/videos/284_263.mp4'

frames = get_frames(video_file, startingPoint=0)
faces = get_faces(frames, height=height, width=width, number_of_faces=number_of_faces)
fig, ax = plt.subplots(1,1, figsize=(15, 15))
ax.imshow(faces[0])
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.title.set_text(f"FRAME 0: {video_file.split('/')[-1]}")
plt.grid(False)
plt.show()
plt.style.use('ggplot')
train_dir = '/Users/jatinarora/deepfake-detection-challenge/train_sample_videos/'
train_video_files = [train_dir + x for x in os.listdir(train_dir)]
# video_file = train_video_files[30]

## Getting the first frame
# cap = cv.VideoCapture(video_file)
# success, image = cap.read()
# image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# cap.release()
#
# face_locations = face_recognition.face_locations(image)
# print("I found {} face(s) in this photograph.".format(len(face_locations)))
#
# for face_location in face_locations:
#
#     # Print the location of each face in this image
#     top, right, bottom, left = face_location
#     print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
#
#     # You can access the actual face itself like this:
#     face_image = image[top:bottom, left:right]
#     fig, ax = plt.subplots(1,1, figsize=(5, 5))
#     plt.grid(False)
#     ax.xaxis.set_visible(False)
#     ax.yaxis.set_visible(False)
#     ax.imshow(face_image)
#     plt.show()
#
# face_landmarks_list = face_recognition.face_landmarks(image)
# pil_image = Image.fromarray(image)
# d = ImageDraw.Draw(pil_image)
#
# for face_landmarks in face_landmarks_list:
#
#     # Print the location of each facial feature in this image
#     for facial_feature in face_landmarks.keys():
#         print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))
#
#     # Let's trace out each facial feature in the image with a line!
#     for facial_feature in face_landmarks.keys():
#         d.line(face_landmarks[facial_feature], width=3)
#
# # Show the picture
# pil_image.show()