import numpy as np
import cv2
from matplotlib import pyplot as plt
import json

image1 = cv2.imread("gambar/databaru/gb1.jpg")
gray_image1= cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2 = cv2.imread("gambar/databaru/gb2.jpg")
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
image3 = cv2.imread("gambar/databaru/gb3.jpg")
gray_image3= cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
image4 = cv2.imread("gambar/databaru/gb4.jpg")
gray_image4= cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)
image5 = cv2.imread("gambar/databaru/gb5.jpg")
gray_image5= cv2.cvtColor(image5, cv2.COLOR_BGR2GRAY)
image7 = cv2.imread("gambar/databaru/gb7.jpg")
gray_image7= cv2.cvtColor(image7, cv2.COLOR_BGR2GRAY)

print(image1[0][0])
print(image1[0][1])
print(image1[0][2])
print(image1[0][3])
print(image1[0][4])
print(gray_image1[0])

temp1 = []

for i in range(len(gray_image1)):
    for j in range(len(gray_image1[0])):
        temp1.append([gray_image1[i][j],gray_image2[i][j],gray_image3[i][j],gray_image4[i][j],gray_image5[i][j],gray_image7[i][j]])
Z = temp1
Z = np.float32(Z)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(Z,5,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

A = Z[label.ravel()==0]
B = Z[label.ravel()==1]
C = Z[label.ravel()==2]
D = Z[label.ravel()==3]
E = Z[label.ravel()==4]
cv2.TERM_CRITERIA_EPS

x=0
y=0
for i in range(len(label)):
    if label[i]==1:
        image1[x][y]=[255,0,0]
    elif label[i]==2:
        image1[x][y]=[0,255,0]
    elif label[i]==3:
        image1[x][y]=[0,0,255]
    elif label[i]==4:
        image1[x][y]=[255,255,0]
    else:
        image1[x][y]=[0,255,255]

    if(y<len(image1[0])-1):
        y=y+1
    else:
        y=0
        x=x+1
# cv2.imshow('image',image1)
# cv2.waitKey(0)
cv2.imwrite('gambar/databaru/hasil.png',image1)
# Plot the data
plt.scatter(A[:,0],A[:,1],c = 'm')
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(C[:,0],C[:,1],c = 'b')
plt.scatter(D[:,0],D[:,1],c = 'y')
plt.scatter(E[:,0],E[:,1],c = 'g')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'c', marker = 's')
plt.xlabel('X'),plt.ylabel('Y')
plt.savefig('gambar/cluster.jpg')
# plt.show()