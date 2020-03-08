
import cv2 as cv
import face_recognition
import math


def get_frames(video_path,number_of_frames=1):
    cap = cv.VideoCapture(video_path)
    images=[]
    for i in range(1,number_of_frames):
        success, image = cap.read()
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        images.append(image)
    cap.release()
    return images

def get_faces(frames,height=-1,width=-1):
    face_images = []
    for i in range(1, len(frames)):
        face_locations = face_recognition.face_locations(frames[i])
        for face_location in face_locations:
            top, right, bottom, left = face_location

            if (height!=-1 and (bottom-top) < height):
                diff = height - (bottom-top)
                top = top - math.ceil(diff/2)
                bottom = bottom + math.floor(diff/2)
            if (height != -1 and (bottom - top) > height):
                diff = bottom-top-height
                top = top + math.floor(diff/2)
                bottom = bottom - math.ceil(diff/2)

            if(width!=-1 and (right-left)<width):
                diff = width -(right-left)
                left = left - math.ceil(diff/2)
                right = right+math.floor(diff/2)

            if (width != -1 and (right-left) > width):
                diff = (right - left) - width
                left = left + math.ceil(diff / 2)
                right = right - math.floor(diff / 2)

            face_image = frames[i][top:bottom, left:right]
            face_images.append(face_image)
    return face_images





