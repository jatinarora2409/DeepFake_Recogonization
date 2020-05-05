# Deepfake Detection using Deep learning 

## Presentation: 
Slides: https://docs.google.com/presentation/d/1kXOYvH12RVMsDP70LUsyE3m3h8UmlLjmGQAuOlr9aPs/edit?usp=sharing <br />
Video Presentation: https://github.com/jatinarora2409/DeepFake_Recogonization/blob/master/presentation.mp4

## Project Goal

**To explore different computer vision methods coupled with deep learning networks to find patterns and inconsistencies in the deepfake videos**

* Train and compare Deep Neural Networks with different pre-trained convolutional layers using Transfer learning.


* Explore new features which can be used to detect inconsistencies in videos.


* Evaluate if there is any temporal information embedded in videos that can help identify deep fakes

* Determine if LSTMs can sniff out any signal from multiple frames of a video

* Evaluate if noise pattern in images is a viable parameter in detecting forgeries 


## Problem Statement
Deepfakes are videos and images which are generated using the combination of autoencoders and generative adversarial networks. These images and videos are synthetic videos which look highly similar to the real videos. Specifically for celebrities and politicians, who have a high number of images available on the internet, creating deepfakes has become a very easy process using applications available on the internet.  With the improvement in  GANs and autoencoders,  deepfakes are becoming very difficult to detect with human eyes and even by the state-of-art deepfake detection techniques. It is not difficult to list down the dangers of deepfake videos and images. With the spreading of false compelling aggressive speech by popular world figures, it can easily become an issue to world security leading to civil unrest and damaging public image. In many cases, videos and images are produced as evidence in matters of law. Deepfakes if not detected, can induce false bias in law. Other than the above two examples,  many other negative impacts of deepfake like false images for military analysis exists. We will concentrate on a subset of videos that involve manipulation of the actors’ faces, which can either be expression manipulation or identity manipulation.  Faces are of special interest since they play a central role in human communication as the face of a person can emphasize a message or it can even convey a message just in itself.

## Why is the problem motivating and hard?

To see an example: https://www.youtube.com/watch?v=fqye4Cw0EF0 is a deepfake from star trek seeing, with actors replaced with Jeff Bezos and Elon Musk fakes. Who can say, this is a deepfake video? 

**Seeing is believing is no more true.** We are not able to trust anything we see or hear. Today, deepfakes are becoming so natural and strong that it is next to impossible to completely detect forgery with machines, leave alone naked eyes.  The thing to worry about -  " Theoretically if you gave a GAN all the techniques we know to detect it, it could pass all of those techniques. We don't know if there's a limit. It's unclear. " statement to MIT Technology Revie by DARPA Program manager  At the end of the day, the hype around deep fakes may be the greatest protection we have. The fact that we know, a video can be forged can protect us from trusting any video.

We believe that it is only a matter of time before it will become impossible to detect deepfakes. After all, a video is a set of pixels/numbers. 



## How are deepfakes Generated? 
![](https://github.com/jatinarora2409/DeepFake_Recogonization/blob/master/deepfake.png)
Image Reference : https://engineering.purdue.edu/~dgueraco/content/deepfake.pdf

The deep fakes video generation is a 2 step process. 

1. **Training 2 Autoencoders** : 
Two autoencoders are trained with 2 datasets, one set of data of person say A in the video, and one says B with whose's face the A will be replaced. Encoders share the same parameters, so they take note of the same features. The decoders are trained separately.  **The encoders as they have the same parameters will note similar features of the face.**


2. **Generation** :
Once we have the encoder and both the decoders, we use the face from the video frame, encode it using the common encoder and decode with the decoder of face A. This will replace the face B, with a face looking highly similar to A. 


## State of the art 
With the popularity gained by deepfakes, it has attracted many researchers and top companies with likes of Google, Facebook, Amazon.

1. A good amount of research in the last two-three years has been focused on the facial artifacts of a person’s face. Lead-ing Research in UC Berkeley and USC is on the same lines, recognizing the speaking and movement pattern of a single in-dividual from her true videos and comparing the model with questionable videos to check its authenticity. 
They do not aim to generalize the solution. The research is based on learning the moment of a single person. This protects a specific person. 
Reference: https://www.ischool.berkeley.edu/news/2019/uc-berkeley-researchers-creating-deepfake-detection-software-newsrooms

2. A Kaggle competition on deepfake detection has just ended. The results will probably reveal a new SOA in a few days
Current SOA are all Deep Neural Networks. Most of the manipulated videos are compressed, which tends to obscure any artifacts leftover by the manipulation. Standard Image forensics techniques yield very poor results. ‘SOA’ for the FaceForensics++ dataset is 94%, but models provided by them don’t actually perform that good. Fine-tuning to reach that performance not revealed by them.
Reference: https://www.kaggle.com/c/deepfake-detection-challenge

## Dataset

For our project, we used the dataset from FaceForencis
Reference: https://github.com/ondyari/FaceForensics/tree/master/dataset 

The dataset includes 977 downloaded videos from youtube, 1000 original extracted sequences that contain an unobstructed face that can be easily tracked, as well as their manipulated versions using our four methods: Deepfakes, Face2Face, FaceSwap, and NeuralTextures. **For our experiments, we used the original video sequences, and the deepfake method created dataset.**

## Experiments

For our experiments, we explored various deep learning models. We used the Convolutional Neural Network as well as 
Recurrent Neural Network. 

We used the face-recognition library to find the faces in the frames and cropped them. 
Reference: https://github.com/ageitgey/face_recognition
![](https://github.com/jatinarora2409/DeepFake_Recogonization/blob/master/deepfake_3.png)

We used OpenCV library for Laplacian filters, normalization, etc. 
Reference https://pypi.org/project/opencv-python/

Below is the list of all models, we have run our experiments on. <br />

### Models Tested 

1. **CNN** -  The model consists of 3 layers, 64 filters, 32 filters, and an output layer of 2 nodes. Different features used with the CNN model <br />
a. Cropped Faces Frames<br />
b. Laplacian Filter Frames<br />
 
2. **RNN** - **We used the model ResNet50.** We tested with the  feature of a sequence of 40 frames Cropped Faces from each video

3. **CNN + LSTM**: This the model we created ourselves without any pre-trained weights and tested with various features. The model consisted of a 24,48,64 layer CNN layer, 64 LSTM Cells connected to 1024,512 nodes 2 Fully connected layers followed by 2 output nodes. <br />

We used various features: <br />
a. Cropped Faces Frame Sequences<br />
b. Laplacian Filter <br />

![](https://github.com/jatinarora2409/DeepFake_Recogonization/blob/master/deepfake_2.png)
Image Reference: https://engineering.purdue.edu/~dgueraco/content/deepfake.pdf


4. **InceptionV3 + 2 Dense Layers**: We used a pre-trained model on ImageNet called InceptionV3 to extract a 2048 length vector from each image frame. We tested the model with the Cropped Sequence of Face Frames with normalization. The 2 dense layers include a 2048 unit layer with ReLu activation and a 12 unit layer with sigmoid activation.
 

5. **Xception + 2 Dense Layers:** We used a pre-trained model on ImageNet called Xception to extract a 2048 length vector from each image frame. We tested the model with the Cropped Sequence of Face Frames with normalization. The 2 dense layers include a 2048 unit layer with ReLu activation and a 12 unit layer with sigmoid activation.

6. **FaceNet + 2 Dense Layers:** We used a pre-trained model on LFW called FaceNet to extract a 128 length vector from each image frame. We tested the model with the Cropped Sequence of Face Frames with normalization. The 2 dense layers include a 128 unit layer with ReLu activation and a 10 unit layer with sigmoid activation.


## Results

**Basic 3 layer CNN RNN Results**: 
We trained the models on 300 videos with 80 frames from each video ( 24000 images/frames). The accuracy for both the models hovered around 53-55%. 

**CNN + LSTM Result:** 
We trained the model on 300 videos with 40 frames from each video (12000 images/frames). First, we trained on cropped faces sequences and laplacian filter <br />
For Cropped Faces, the accuracy was limited to _53%_
With Laplacian Filter, the accuracy increased to _60%_<br />
*NOTE: Laplacian filter can be handy in creating feature vectors when detecting frames for forgery* 
 
**InceptionV3 + 2 Dense Layers Results:**
The model over-fits the training set of 24000 images presumably because of the larger capacity of the dense layers. The models in-fact over-fit all the three training scenarios which are expected given our limited training set.
* **Accuracy: 75%**
* **Precision: 66%**
* **Recall: 82%**

**Xception + 2 Dense Layers Results:** 
* **Accuracy: 72%**
* **Precision: 65%**
* **Recall: 71%**

**FaceNet + 2 Dense Layers Results**
* **Accuracy: 82%**
* **Precision: 74%**
* **Recall: 89%**

## Analysis 
FaceNet seems to do a better job at classifying deepfakes with Recall being almost 90%. This is in-line with our requirements that false negatives below - it's okay if some originals are classified as forged but a forged video should be caught by the classifier. The Advantage here seems to be coming from the fact that FaceNet is trained exclusively on images of faces whereas the other two used ImageNet for training which encompasses all categories of images.  
All three models over-fit to almost 100% accuracy on the training data within 20 epochs but the validation accuracy surprisingly doesn't go down. This hints at a large untapped capacity of the model and that we're under-feeding the model during training. The accuracy of all the three models should go up either by adding more videos to the training or increasing the number of frames from each video.  
<table>
 <tr>
    <td> <img src="https://github.com/jatinarora2409/DeepFake_Recogonization/blob/master/deepfakeresults/IMG-20200504-WA0044.jpg" alt="Drawing" style="width: 100px;"/> </td>
    <td> <img src="https://github.com/jatinarora2409/DeepFake_Recogonization/blob/master/deepfakeresults/IMG-20200504-WA0043.jpg" alt="Drawing" style="width: 100px;"/> </td>
    </tr>
</table>  
The left image is sourced from the original video and the right from the forged video. The forged image is fed to the model and it correctly predicts that it is a fake with probability 0.98.  
  

<table>
 <tr>
    <td> <img src="https://github.com/jatinarora2409/DeepFake_Recogonization/blob/master/deepfakeresults/IMG-20200504-WA0045.jpg" alt="Drawing" style="width: 100px;"/> </td>
    <td> <img src="https://github.com/jatinarora2409/DeepFake_Recogonization/blob/master/deepfakeresults/IMG-20200504-WA0040.jpg" alt="Drawing" style="width: 100px;"/> </td>
    </tr>
</table>  

The above is an example of a false negative, the model predicts that the image on the right is not forged (0.23). On visual analysis, the fake does indeed look convincing and thus confused our rather weakly trained model.

  

<table>
 <tr>
    <td> <img src="https://github.com/jatinarora2409/DeepFake_Recogonization/blob/master/deepfakeresults/IMG-20200504-WA0046.jpg" alt="Drawing" style="width: 100px;"/> </td>
    <td> <img src="https://github.com/jatinarora2409/DeepFake_Recogonization/blob/master/deepfakeresults/IMG-20200504-WA0047.jpg" alt="Drawing" style="width: 100px;"/> </td>
    </tr>
</table>  

The model rightly predicts that the image on the right is forged (0.87). It looks like the model on-par with a human observer in identifying forgeries.  
  

<table>
 <tr>
    <td> <img src="https://github.com/jatinarora2409/DeepFake_Recogonization/blob/master/deepfakeresults/IMG-20200504-WA0042.jpg" alt="Drawing" style="width: 100px;"/> </td>
    <td> <img src="https://github.com/jatinarora2409/DeepFake_Recogonization/blob/master/deepfakeresults/IMG-20200504-WA0041.jpg" alt="Drawing" style="width: 100px;"/> </td>
    </tr>
</table>  

The model gets this one wrong (0.45) although 0.45 is pretty close to the threshold of 0.5. It also looks like the model is not yet relying on sudden gradients on faces which can be seen in the image on the right. This varying skin tone can be a very useful feature to identify forgeries like these which are rather poorly generated.



## Difficulties and Challenges
* Due to limited Resources with only 1 GPU available, we were able to train on a limited number of videos for most of the models. The training with 1 GPU use to take a lot of time, therefore we couldn't do much hit and trial with various features. 

* The main goal of our project was to explore various to test various features we can find in videos, but most of the time went in setting up the machine learning environment and models.

In future work, we would like to set up a machine learning env online, where researchers can submit scripts to get features out of frames and use those for training learning models. We believe that this product will help FastTrack the research and can be packaged as a product. 

## Conclusions
* Neural Networks tested here are definitely picking up a signal and are able to identify forgeries with decent accuracy.
* We Need a much larger dataset to see >90% accuracy, as reported by the FaceForensics team. The dataset may be too constrained in the type of videos though
* We will need some authentication scheme for videos to catch such forgeries as Deepfakes keep getting better.



