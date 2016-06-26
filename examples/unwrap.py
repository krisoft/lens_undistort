# this is just a quick example to show how one can use the unwrap matrix hdf5 file generated by the lens_undistort tool

import cv2
import numpy as np
import h5py

datastore = h5py.File('unwrap_matrix.hdf5','r')
unwrap_map = datastore["map"]
unwrap_map = np.array(unwrap_map) # unwrap_map is a 'numpy array like' thing, make it an actuall numpy array


input_frame = cv2.imread("test.png")
unwrapped = cv2.remap( input_frame, unwrap_map, None, cv2.INTER_LANCZOS4 )

cv2.imshow("unwrapped", unwrapped)
cv2.waitKey(0)