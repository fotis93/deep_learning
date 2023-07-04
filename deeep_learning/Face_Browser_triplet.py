# first import the module
import webbrowser
import cv2
import os 

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torchvision
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv4 = nn.Conv2d(32, 64, 5)

        #self.fc1 = nn.Linear(64 * 2 * 2, 120)
        #self.fc1 = nn.Linear(64 * 8 * 8, 120)
        self.fc1 = nn.Linear(64 * 15 * 15, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, arg):
        output = self.pool(F.relu(self.conv1(arg)))
        #print(output.size())
        output = self.pool(F.relu(self.conv2(output)))
        #print(output.size())
        output = self.pool(F.relu(self.conv3(output)))
        #print(output.size())
        output = self.pool(F.relu(self.conv4(output)))
        #print(output.size())
        output = torch.flatten(output, 1) # flatten all dimensions except batch
        #print(output.size())
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        return output


vid = cv2.VideoCapture(0)

#video metadata
fps = vid.get(cv2.CAP_PROP_FPS)
print('frames per second =',fps)
count = 0
while(count<10):
	
	# Capture the video frame
	# by frame
	ret, frame = vid.read()
	path = 'deeep_learning/check_photos/person'
	
	pathtowrite = os.path.join(path, "frame%d.jpg" % count)
	count+=1
	cv2.imwrite(pathtowrite, frame)
		

		#print(os.path.join(path, 'val' , "frame%d.jpg" % count) )
		#cv2.imwrite("frame%d.jpg" % count, frame)
	# Display the resulting frame
	cv2.imshow('frame', frame)
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

######loading the network 

Mode = torch.load('deeep_learning/triplet_selection_models/model_67')
Mode = Mode.to(device)
Mode.eval()

#####using the same transformations as in training 
data_transforms = transforms.Compose([
        transforms.Resize(330),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


#this is the directory where i have saved my images 
dir  = 'deeep_learning/check_photos'
image_datasets =  datasets.ImageFolder(dir, data_transforms)
dir2  = 'deeep_learning/IDphotos/train'
target_datasets =  datasets.ImageFolder(dir2, data_transforms)


###i want a 70% succes on the images i took in order to open the web 
count=0
sum =0
dis = 1.7
target = target_datasets[0][0]
vec_t = Mode(target.unsqueeze(0).to(device))
#print(vec_t)
#print(fotis)
for photo,label in image_datasets:
    photo2 = photo.unsqueeze(0).to(device)
    res = Mode(photo2)
    v = res-vec_t
    dist = torch.matmul(v, v.t())
    if dist<dis*dis:
         sum+=1
    count+=1
    print(sum)


###putting the threshold to open the camera 
threshold = 0.7
if sum >=threshold*count:
    print("Welcome Master")
    # then make a url variable
    url = "https://store.steampowered.com/"
    
    # then call the get method to select the code 
    # for new browser and call open method 
    # described above
    webbrowser.open(url)
    
    # results in error since chrome is not registered initially.
else:
    print("You are not authorized to access the web")
	

