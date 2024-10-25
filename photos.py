import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import numpy as np

DIRECTORY = "faces/"
CATEGORIES = os.listdir(DIRECTORY)

data = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    label = CATEGORIES.index(category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        if img_array.shape[:2] < (50, 50):
            break
        img_array = cv2.resize(img_array, (100, 100))
        data.append([img_array, label])


random.shuffle(data)
print(len(data))

x = []
y = []


for features, labels in data:
    x.append(features)
    y.append(labels)


x = np.array(x)
y = np.array(y)

pickle.dump(x, open("X.pk1", "wb"))
pickle.dump(y, open("Y.pk1", "wb"))