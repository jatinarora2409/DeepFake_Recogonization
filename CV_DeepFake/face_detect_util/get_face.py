
import cv2 as cv
import face_recognition
import math
import numpy as np
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage import  img_as_float
from skimage.util import random_noise

sigma = 0.12

def get_frames(video_path,number_of_frames=-1,startingPoint=0):
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
    print(" From Video File " + str(video_path) )
    return images

def get_faces(frames,height=-1,width=-1,number_of_faces=-1):
    face_images = []
    collected_faces = 0

    for i in range(0, len(frames)):
        if(collected_faces==number_of_faces):
            break;
        face_locations = face_recognition.face_locations(frames[i])
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = frames[i][top:bottom, left:right]
            dsize = (width, height)
            face_image = cv.resize(face_image,dsize)
            original = img_as_float(face_image)

            noisy = random_noise(original, var=sigma ** 2)
            sigma_est_true = estimate_sigma(noisy, multichannel=True, average_sigmas=True)
            fixed_noisy_true = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
                                               method='VisuShrink', mode='soft',
                                               sigma=sigma_est_true / 4, rescale_sigma=True)
            photo = original - fixed_noisy_true
            face_image = np.array(photo)
            face_images.append(face_image)
            collected_faces = collected_faces+1;
            ## TODO: Limiting to one face per video
            break;
    print("Addded These many faces ", len(face_images))
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

 # if (height!=-1 and (bottom-top) < height):
            #     diff = height - (bottom-top)
            #     top = top - math.ceil(diff/2)
            #     bottom = bottom + math.floor(diff/2)
            # if (height != -1 and (bottom - top) > height):
            #     diff = bottom-top-height
            #     top = top + math.floor(diff/2)
            #     bottom = bottom - math.ceil(diff/2)
            #
            # if(width!=-1 and (right-left)<width):
            #     diff = width -(right-left)
            #     left = left - math.ceil(diff/2)
            #     right = right+math.floor(diff/2)
            #
            # if (width != -1 and (right-left) > width):
            #     diff = (right - left) - width
            #     left = left + math.ceil(diff / 2)
            #     right = right - math.floor(diff / 2)
            #
            # #print("Frame Limits:" + str(frames[i].shape))
            # max_height, max_width,channels  = (frames[i].shape)
            #
            # if(top<0):
            #     bottom = bottom - top;
            #     top = 0
            # if(bottom>=max_height):
            #     top = top - (bottom-max_height+1)
            #     bottom = max_height-1
            # if(left<0):
            #     right = right - left
            #     left = 0
            # if(right>max_width):
            #     left = left-(right-max_width+1)
            #     right = max_width-1
            #
            # face_image = frames[i][top:bottom, left:right]