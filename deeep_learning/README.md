# deep_learning

videotoframes.py : A short python script to get face data from the camera for the training

transfer_learning-custom_cnn.py : This file containts the two CNN networks, it trains them and produces graphs for each

triplet_loss_online_training.py: This trains a triplet loss model of two classes by online semi hard triplets as in FaceNet

Conditional Triplet Loss for Few-shot Learning.py : This trains a triplet loss model with the methods described in A Conditional Triplet Loss for Few-shot Learning and its Application to Image Co-Segmentation

Face_Browser.py: This opens a browser when the transfer learning CNN trained in transfer_learning-custom_cnn.py identifies the person in question

Face_Browser_triplet.py : THe same as Face_Browser.py but with triplets, it is not as reliable 

check_photos: The directory where the phots taken during the Face_Browser or Face_Browser_triplet are saved

graphs : THe directory the comparative graphs between the custom CNN and transfer learning model are stored 

IDnetwork : THe directory where the custom CNN and transfer learning are stored after training to be used later.

IDphotos : The directory where the photos that will be used to train the models are stored 

model_Data :  A file that contains results of the two triplets models

prc_curves_custom_cnn,prc_curves_transfer_learning,roc_curves_custom_cnn,roc_curves_transfer_learning = files that containt roc and prc curves created from the transfer_learning-custom_cnn.py that are about the training and testing of the two CNN models

triplet_random_models,triplet_selection_models = models that contain the saved Triplet loss models after training 

test_for_triplet.py = Python script that creates the results for the triplet loss models 
