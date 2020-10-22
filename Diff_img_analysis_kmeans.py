#Harish Bachu

#Python Implementation

#Paper on Unsupervised Change Detection in Satellite Images
#Using Principal Component Analysis
#and k -Means Clustering

#07 May 2020

'''
STEPS:
1. CREATE DIFFERENCE IMAGE X_d = |X1 - X2|
2. CREATE hxh NON-OVERLAPPING BLOCKS (h > 2)
3. GREATE EIGENVECTOR SPACE USING PCA
4. GENERATE 2 CLUSTERS FROM FEATURE VECTOR SPACE USING K-MEANS
5. ASSIGN EACH FEATURE VECTOR TO THE NEAREST CLUSTER\
'''

import numpy as np 
import cv2
from matplotlib import pyplot as plt 
from math import gcd
from sklearn.cluster import KMeans

rate = 0.9

def get_blocks(X_d, H, W, h):							#Returns Vectors

	blocks = []
	for i in range(H):
		for j in range(W):
			blocks.append(np.reshape(X_d[i : i + h, j : j + h], (h**2, 1)))
	blocks = np.array(blocks)
	return blocks


def get_square(blocks):									#Returns Square for generating C

	squares = []
	for i in blocks[:5]:
		j = np.array([i])
		squares.append(np.dot(j.T, j))
	return np.array(squares)


def PCA(C, blocks):										#Performs PCA

	C_eigenvalue, C_eigenvector = np.linalg.eig(C)
	ids = C_eigenvalue.argsort()[::-1]
	C_eigenvalue = C_eigenvalue[ids]
	C_eigenvector = C_eigenvector[ids]
	for i in range(len(C_eigenvalue)):
		if sum(C_eigenvalue[i:]) >= rate*sum(C_eigenvalue):
			break
	eigenvector = C_eigenvector[:, i:]
	feature_vector = np.dot(blocks, eigenvector)
	return feature_vector


def K_means(feature, H, W):								#K-Means Clustering for Change Detection

	label = KMeans(2).fit(feature).labels_
	c_map = np.reshape(label, (H, W))					# STEP 5
	if sum(sum(c_map == 1)) > sum(sum(c_map == 0)):
		c_map[c_map == 1] = 0
		c_map[c_map == 0] = 1
	c_map = np.abs(c_map)*255
	c_map = c_map.astype(np.uint8)
	return c_map


def get_change_map(X1, X2):

	rows, cols = X1.shape
	X1 = cv2.resize(X1, (cols, rows))
	X2 = cv2.resize(X2, (cols, rows))
	
	X_d = cv2.absdiff(X1, X2)							# STEP 1
	H, W = X_d.shape
	
	h = 3												#block size

	pads = int(np.ceil(h / 2))							
	X_d = np.pad(X_d, ((pads, pads), (pads, pads)))		#padding difference image

	blocks = get_blocks(X_d, H, W, h)[:,:,0]			# STEP 2
	M = int(W*H/h**2)									#M

	C = np.sum(get_square(blocks), axis = 0)/M 			#C (h^2 x h^2) matrix for eigendecomposition
	
	feature = PCA(C, blocks)							# STEP 3
	
	change_map = K_means(feature, H, W)					# STEP 4
	
	return X_d, change_map

if __name__ == "__main__":

	X1 = cv2.imread('/home/harishbachu/Documents/ISRO/corpus/AndasolChangePair/Andasol_09051987_md.jpg', 0)
	X2 = cv2.imread('/home/harishbachu/Documents/ISRO/corpus/AndasolChangePair/Andasol_09122013_md.jpg', 0)
	difference, change = get_change_map(X1, X2)
	cv2.imshow("Image 1", X1)
	cv2.imshow("Image 2", X2)
	cv2.imshow("Difference Image", difference)
	cv2.imshow("Change", change)
	cv2.waitKey(0)
	cv2.destroyAllWindows()