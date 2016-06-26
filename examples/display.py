import cv2
import numpy as np
import h5py

import elas

datastore = h5py.File('../data/rectification_01.hdf5','r')
datastore_left = h5py.File('../data/viktor_unwrap.hdf5','r')
datastore_right = h5py.File('../data/aero_unwrap.hdf5','r')


size_left = datastore_left["map"].shape[0:2]
size_right = datastore_right["map"].shape[0:2]

if size_left[0]*size_left[1] < size_right[0]*size_right[1]:
	size = size_right
else:
	size = size_left


blank_image = np.zeros((size[0],size[1],3), np.uint8)
blank_image[:,:] = (255,255,255)
blank_image[::50,:] = (0,0,255)
blank_image[:,::50] = (0,0,255)
blank_image[-1,:] = (0,0,255)
blank_image[:,-1] = (0,0,255)

"""
[u'E', u'F', u'P_left', u'P_right', u'Q', u'R', u'R_left', u'R_right', 
u'T', u'camera_left', u'camera_right', u'map_left', u'map_right', u'mask_left', u'mask_right']
"""

camera_left = np.array(datastore["camera_left"])
dist_left = np.array([0.,0.,0.,0.])
R_left = np.array(datastore["R_left"])
P_left = np.array(datastore["P_left"])

map_left, _ = cv2.initUndistortRectifyMap(
		camera_left, dist_left,
		R_left, P_left,
		(size[1],size[0]), cv2.CV_32FC2
)

rectified_left = cv2.remap( blank_image, map_left, None, cv2.INTER_LANCZOS4 )




camera_right = np.array(datastore["camera_right"])
dist_right = np.array([0.,0.,0.,0.])
R_right = np.array(datastore["R_right"])
P_right = np.array(datastore["P_right"])

map_right, _ = cv2.initUndistortRectifyMap(
		camera_right, dist_right,
		R_right, P_right,
		(size[1],size[0]), cv2.CV_32FC2
)

rectified_right = cv2.remap( blank_image, map_right, None, cv2.INTER_LANCZOS4 )





cv2.imshow("rectified_left", rectified_left )
cv2.imshow("rectified_right", rectified_right )
while True:
	if cv2.waitKey(10)!=-1:
		break