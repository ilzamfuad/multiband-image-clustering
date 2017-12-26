import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
from matplotlib import pyplot as plt
import json
import sys

image1 = cv2.imread("proses/gambar/databaru/gb1.jpg")
gray_image1= cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2 = cv2.imread("proses/gambar/databaru/gb2.jpg")
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
image3 = cv2.imread("proses/gambar/databaru/gb3.jpg")
gray_image3= cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
image4 = cv2.imread("proses/gambar/databaru/gb4.jpg")
gray_image4= cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)
image5 = cv2.imread("proses/gambar/databaru/gb5.jpg")
gray_image5= cv2.cvtColor(image5, cv2.COLOR_BGR2GRAY)
image7 = cv2.imread("proses/gambar/databaru/gb7.jpg")
gray_image7= cv2.cvtColor(image7, cv2.COLOR_BGR2GRAY)


temp1 = []
cluster = int(sys.argv[1])
print(cluster)
for i in range(len(gray_image1)):
    for j in range(len(gray_image1[0])):
        temp1.append([gray_image1[i][j],gray_image2[i][j],gray_image3[i][j],gray_image4[i][j],gray_image5[i][j],gray_image7[i][j]])
Z = temp1
Z = np.float32(Z)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(Z,cluster,None,criteria,30,cv2.KMEANS_RANDOM_CENTERS)

A = Z[label.ravel()==0]
B = Z[label.ravel()==1]
C = Z[label.ravel()==2]
D = Z[label.ravel()==3]
E = Z[label.ravel()==4]
F = Z[label.ravel()==5]
cv2.TERM_CRITERIA_EPS

x=0
y=0
for i in range(len(label)):
    if label[i]==0:
        image1[x][y]=[0,255,255] #bgr
    elif label[i]==1:
        image1[x][y]=[0,0,255]
    elif label[i]==2:
        image1[x][y]=[255,255,0]
    elif label[i]==3:
        image1[x][y]=[255,0,0]
    elif label[i]==4:
        image1[x][y]=[0,255,0]
    elif label[i]==5:
        image1[x][y]=[255,0,255]

    if(y<len(image1[0])-1):
        y=y+1
    else:
        y=0
        x=x+1
# cv2.imshow('image',image1)
# cv2.waitKey(0)
cv2.imwrite('proses/gambar/databaru/hasil.png',image1)
# Plot the data
plt.scatter(A[:,0],A[:,1],c = 'y')
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(C[:,0],C[:,1],c = 'c')
plt.scatter(D[:,0],D[:,1],c = 'b')
plt.scatter(E[:,0],E[:,1],c = 'g')
plt.scatter(F[:,0],F[:,1],c = 'm')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'k', marker = 's')
plt.xlabel('X'),plt.ylabel('Y')
plt.savefig('proses/gambar/cluster.png')
# plt.show()