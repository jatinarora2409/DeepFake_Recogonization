
import cv2 as cv
import face_recognition
import math

import matplotlib.pylab as plt
import numpy as np

def get_frames(video_path,number_of_frames=1,startingPoint=0):
    cap = cv.VideoCapture(video_path)
    cap.set(1,startingPoint)
    images=[]
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if(number_of_frames==-1):
        number_of_frames=100000000000
    for i in range(0,min(number_of_frames,length)):
        success, image = cap.read()
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        images.append(image)
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # plt.grid(False)
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        # ax.imshow(image)
        # plt.show()
    cap.release()
    return images

def get_faces(frames,height=-1,width=-1):
    face_images = []

    for i in range(0, len(frames)):
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

            max_height, max_width  = len(frames[i])
            print("Frame Limits:"+str(max_height,max_width))
            if(top<0):
                bottom = bottom - top;
                top = 0
            if(bottom>=max_height):
                top = top - (bottom-max_height+1)
                bottom = max_height-1
            if(left<0):
                right = right - left
                left = 0
            if(right>max_width):
                left = left-(right-max_width+1)
                right = max_width-1

            face_image = frames[i][top:bottom, left:right]
            face_image = np.array(face_image)
           ##################

            print("top: "+str(top) + " bottom: "+str(bottom) + " left: "+str(left)+" right "+str(right))

            if(bottom-top<height):
                np.pad(face_image,((0,height-(bottom-top)),(0,0)),'constant')
            if(right-left<height):
                np.pad(face_image,((0,0),(0,width-(right-left))),'constant')

            ####################
            print("Face_image_size: "+str(face_image.shape))
            face_images.append(face_image)
            ## TODO: Limiting to one face per video
            break;
    return np.asarray(face_images)


def get_cropped_images(frames,height,width):
    images = []
    for i in range(0, len(frames)):
        frame = frames[i];
        rows,cols,channels = frame.shape
        dist_height = height/2
        top = math.ceil(rows/2)-math.ceil(dist_height)
        bottom = math.ceil(rows/2)+math.floor(dist_height)
        dist_width = width/2
        left = math.ceil(cols/2)-math.ceil(dist_width)
        right = math.ceil(cols/2)+math.floor(dist_width)
        images.append(frame[top:bottom,left:right])

    return images

