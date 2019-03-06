import numpy as np
import cv2
import os
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import random
from keras.optimizers import Adam
from model_zaragoza import Model
from keras.models import model_from_json
image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)
def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


images = []
Bin = [] 
Dir = "/home/sgeadmin/vishay/aip_project/"
inputDir = Dir + "input"
gtDir = Dir + "gt"
orgimagepaths = sorted(list(list_images(inputDir)))
binimagepaths = sorted(list(list_images(gtDir)))
for imagePath in orgimagepaths:
    images.append(cv2.imread(imagePath))
    
images = np.array(images)      

for imagePath in binimagepaths:
    Bin.append(cv2.imread(imagePath,0))
    
Bin = np.array(Bin)
print(Bin.shape)

Imgs = images[:10]
Bins = Bin[:10]
json_file = open('/home/sgeadmin/vishay/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("/home/sgeadmin/vishay/model.h5")
print("Loaded model from disk")

pred = []
idx = 1
for i in range(len(Imgs)):
	aa = np.zeros((np.shape(Imgs[i])[0],np.shape(Imgs[i])[1]))
	for p in range(17,np.shape(Imgs[i])[0]-17):
		for q in range(17,np.shape(Imgs[i])[1]-17):
		    image = np.expand_dims(Imgs[i][p-8:p+9,q-8:q+9], axis=0)
		    temp = model.predict(image)[0]
		    print(temp) 
		    aa[p,q] = np.argmax(temp)*255  
		    pred.append(np.argmax(temp)*255 )
	if idx<10:
        	output_file = "/home/sgeadmin/vishay/aip_project/bincnn/0"+str(idx)+".png"
    	else:
        	output_file = "/home/sgeadmin/vishay/aip_project/bincnn/"+str(idx)+".png"
    	cv2.imwrite(output_file, aa)
    
    	idx+=1 
    	print(idx)

	    	    
	target = Bins[i][17:np.shape(binstest)[0]-17,17:np.shape(binstest)[1]-17].flatten()
	pred = np.array(pred)
	res.append(f1_score(target, pred, average='macro'))
res = np.array(res)
print(np.mean(res))
print(np.std(res))	

