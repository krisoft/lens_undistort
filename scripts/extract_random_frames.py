import cv2
import argparse
import random

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--video')
	parser.add_argument('--output')

	args = parser.parse_args()

	cap = cv2.VideoCapture(args.video)

	frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

	random.seed(42)

	for idx in range(500):
		frame_idx = random.randint(1,frame_count)
		print idx, frame_idx

		cap.set( cv2.CAP_PROP_POS_FRAMES, frame_idx )
		success, frame = cap.read()
		assert success

		cv2.imwrite(args.output+"/%04i.png"%(idx,), frame)