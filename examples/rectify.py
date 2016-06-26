import cv2
import numpy as np
import h5py

import elas

datastore = h5py.File('../data/rectification_01.hdf5','r')

left_unwrap_map = np.array(datastore["map_left"])
right_unwrap_map = np.array(datastore["map_right"])

left_cap = cv2.VideoCapture("/Users/krisoft/progz/own/stereo_calib/data/01/viktor_01.MP4")
right_cap = cv2.VideoCapture("/Users/krisoft/progz/own/stereo_calib/data/01/aero_01.MP4")

frame_count = left_cap.get(cv2.CAP_PROP_FRAME_COUNT)


params = elas.Elas_parameters()
params.postprocess_only_left = False

params.disp_min              = 0;
params.disp_max              = 255;
params.support_threshold     = 0.85;
params.support_texture       = 10;
params.candidate_stepsize    = 5;
params.incon_window_size     = 5;
params.incon_threshold       = 5;
params.incon_min_support     = 5;
params.add_corners           = False
params.grid_size             = 20;
params.beta                  = 0.02;
params.gamma                 = 3;
params.sigma                 = 1;
params.sradius               = 2;
params.match_texture         = 1;
params.lr_threshold          = 2;
params.speckle_sim_threshold = 1;
params.speckle_size          = 200;
params.ipol_gap_width        = 3;
params.filter_median         = False
params.filter_adaptive_mean  = True
params.subsampling           = False


elas_processor = elas.Elas(params)

def nanavg(frame):
	nanmask = np.copy(frame)
	nanmask[nanmask!=np.nan] = 1
	nanmask[nanmask==np.nan] = 0

	return np.nansum(frame, axis=1).astype(np.float32) / np.sum(nanmask, axis=1)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

left_idx = 5.8*60*240 + 15

right_idx = left_idx + 960


left_cap.set( cv2.CAP_PROP_POS_FRAMES, left_idx )
success, left_frame = left_cap.read()
if not success:
	print "no more left"
	exit(-1)

right_cap.set( cv2.CAP_PROP_POS_FRAMES, right_idx )
success, right_frame = right_cap.read()
if not success:
	print "no more right"
	exit(-1)

left_frame = cv2.remap( left_frame, left_unwrap_map, None, cv2.INTER_LANCZOS4 )
right_frame = cv2.remap( right_frame, right_unwrap_map, None, cv2.INTER_LANCZOS4 )


left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

# pre-processing
left_frame = clahe.apply(left_frame)
right_frame = clahe.apply(right_frame)

left_frame = cv2.bilateralFilter(left_frame,9,25,75)
right_frame = cv2.bilateralFilter(right_frame,9,25,75)
# end of pre-processing


left_d = np.empty_like(left_frame, dtype=np.float32)
right_d = np.empty_like(right_frame, dtype=np.float32)


elas_processor.process_stereo(left_frame, right_frame, left_d, right_d)

vis = np.copy(left_d)
vis[vis<=0.0]=0.0

#maxd = np.nanmax(vis)
#print "maxd:", maxd
vis = vis.astype(np.uint8)




cv2.imshow("left", left_frame)
cv2.imshow("right", right_frame)
cv2.imshow("vis", vis)

cv2.imwrite("left.png", left_frame)
cv2.imwrite("right.png", right_frame)
cv2.imwrite("disparity.png", vis)

while True:
	c = cv2.waitKey(10)
	if c!=-1:
		break