# deep_learning

videotoframes.py : A short python script to get face data from the camera for the training

transfer_learning-custom_cnn.py : This file containts the two CNN networks, it trains them and produces graphs for each

triplet_loss_FaceNet.py: This trains a triplet loss model of two classes by online semi hard triplets as in FaceNet

Conditional Triplet Loss for Few-shot Learning.py : This trains a triplet loss model with the methods described in A Conditional Triplet Loss for Few-shot Learning and its Application to Image Co-Segmentation

Face_Browser.py: This opens a browser when the transfer learning CNN trained in transfer_learning-custom_cnn.py identifies the person in question

Face_Browser_triplet_FaceNet.py : THe same as Face_Browser.py but with triplets this is implemented with the net saved from triplet_loss_FaceNet.py

Face_Browser_triplet_Conditional Triplet Loss.py : THe same as Face_Browser.py but with triplets this is implemented with the net saved from Conditional Triplet Loss for Few-shot Learning.py

papers : This directory contains the papers we used

check_photos: The directory where the phots taken during the Face_Browser or Face_Browser_triplet are saved

graphs : THe directory the comparative graphs between the custom CNN and transfer learning model are stored

IDnetwork : THe directory where the custom CNN and transfer learning are stored after training to be used later.

IDphotos : The directory where the photos that will be used to train the models are stored

model_Data : A file that contains results of the two triplets models

prc_curves_custom_cnn,prc_curves_transfer_learning,roc_curves_custom_cnn,roc_curves_transfer_learning = files that containt roc and prc curves created from the transfer_learning-custom_cnn.py that are about the training and testing of the two CNN models

test_for_triplet.py = Python script that creates the results for the triplet loss models that are saved in model_Data

Some technical difficulties arose. One was the path used in the files. While we tried to use relative paths the virtual environment interpreter from Visual studio code would always start from the user file. We have initiated the paths used in the GitHub repository from the deeep learning file but it is possible to not work so this could need some tweaking. Another possible problem that can happen is the use of the model_weights.pth file in the Face_Browser.py. During the training of the model we used a gpu in order to speed up the training. If the computer loading the model does not have cuda the weights may not load. Finally because in the training the camera of the computer,if used with another camera it could be that its performance is not satisfactory.
