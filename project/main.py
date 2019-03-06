import numpy as np
import cv2
import os
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import random
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from model_zaragoza import Model
model = Model(17,17,3,2).build()


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

imgs = images[:11]
bins = Bin[:11]


samples = []
labels = []
for i in range(len(imgs)):
    for p in range(17,np.shape(imgs[i])[0]-17):
        for q in range(17,np.shape(imgs[i])[1]-17):
            samples.append(imgs[i][p-8:p+9,q-8:q+9])
            labels.append(bins[i][p,q])

a = random.sample(range(len(labels)), 3000000)
X = [samples[i] for i in a]
Y = [[1,0] if labels[i]==0 else [0,1] for i in a ]

EPOCHS = 1
INIT_LR = 1e-3
BS = 32

X = np.array(X, dtype="float") / 255.0
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X,
    Y, test_size=0.25, random_state=42)
print(y_train, y_test)

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BS, verbose=1, validation_split = .1)

t = [np.argmax(i) for i in y_test]

t = np.array(t)

p = [np.argmax(i) for i in model.predict(X_test)]

p = np.array(p)


print(f1_score(t, p, average='macro'))

model_json = model.to_json()
with open("./model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("./model.h5")
print("Saved model to disk")
 
# plot training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

