# import the opencv library
import cv2
import os 


# define a video capture object
vid = cv2.VideoCapture(0)

#video metadata
fps = vid.get(cv2.CAP_PROP_FPS)
print('frames per second =',fps)
count = 0
count2 = 0
while(count<300):
	
	# Capture the video frame
	# by frame
	ret, frame = vid.read()
	
	if ( (count%3) == 0):
		path = 'deeep_learning/IDphotos'

		if (count2!=3):
			pathtowrite = os.path.join(path, 'train' , "frame%d.jpg" % count)
			count2+=1
		else :
			pathtowrite = os.path.join(path, 'val' , "frame%d.jpg" % count)
			count2=0
		cv2.imwrite(pathtowrite, frame)
		

		#print(os.path.join(path, 'val' , "frame%d.jpg" % count) )
		#cv2.imwrite("frame%d.jpg" % count, frame)
	count+=1
	# Display the resulting frame
	cv2.imshow('frame', frame)
	
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
