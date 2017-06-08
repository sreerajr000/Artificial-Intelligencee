import json
import numpy as np
import glob
import matplotlib.image as mpimg
import os
import cv2

with open("label.json") as f:
	data = json.load(f)
print (len(data))

emo_img = np.ndarray(shape=(690,10000),dtype=np.float32)
emo_lab = np.ndarray(shape=(690,10),dtype=np.float32)


def one_hot(i):
        a = np.zeros(10, 'float32')
        a[i] = 1.0
        return a

j = 0
for i in data.keys():
        if data[i] != 10:
                try:
                        directory = "faces"
                        f = i+".jpg"
                        path = os.path.join(directory,f)
                        img = mpimg.imread(path)
                        img = cv2.resize(img,(100,100))
                        img = img.reshape((10000))
                        emo_img[j] = img
                        emo_lab[j] = one_hot(data[i])
                        j = j + 1
                except:
                        pass
print (j)
outfile_i = open("images.npy", "wb")
np.save(outfile_i,emo_img)
outfile_i.close()
outfile_l = open("labels.npy", "wb")
np.save(outfile_l,emo_lab)
outfile_l.close()
