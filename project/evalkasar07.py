import cv2
import numpy as np
import sys
import os.path
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

idx = 1
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
predDir = Dir + "bin07"
gtDir = Dir + "gt"
orgimagepaths = sorted(list(list_images(predDir)))
binimagepaths = sorted(list(list_images(gtDir)))
for imagePath in orgimagepaths:
	temp = cv2.imread(imagePath,0) 
	temp = cv2.bitwise_not(temp)   
	if idx<10:
        	output_file = "/home/sgeadmin/vishay/aip_project/bin07/0"+str(idx)+".png"
    	else:
        	output_file = "/home/sgeadmin/vishay/aip_project/bin07/"+str(idx)+".png"
    	cv2.imwrite(output_file, temp)
    
    	idx+=1 
    	print(idx)
	images.append(temp)
	
    
images = np.array(images)      

for imagePath in binimagepaths:
    Bin.append(cv2.imread(imagePath,0))
    
Bin = np.array(Bin)
res = []
for i in range(len(images)):
	pred = images[i].flatten()
	target = Bin[i].flatten()
	res.append(f1_score(target, pred, average='macro',labels=np.unique(pred)))
res = np.array(res)
print(np.mean(res))
print(np.std(res))
	
    
