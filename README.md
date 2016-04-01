This software finds the undistorsion parameters for your camera lens. I used it with some gopros, where  the usual checkerboard based opencv calibration didn't work well. If you have the same problem this might be the solution for you.

#### General description:
1. Get a planar light coloured plain surface. Draw, print, paint, or tape dark straight stripes on it.
2. Record a video, or make still frames with your camera. The framing should be so that the surface fills the whole picture. (see examples) So the stripes leave the picture at the borders. Move the camera around, also rotate it around the optical axis. Record the stripes in both horizontal, vertical, and various cross positions. Make sure that you have pictures where the lines are around the middle of the frame, to one of the sides, and also where they cross through a corner. Now we know that all of those contours should be straight, but they are not because our camera distorted them. Move the
3. Use the software on these inputs. It will run a non-linear optimalization to find the parameters which transform the curved contours into lines.
4. The software will write these parameters into a file. It will also generate a hdf5 file with the undistortion mapping, which you can use with cv::remap to undistort in your software, whenever you need to.

Take a look at the example pictures. The real problem with the "bad.png" is not that it's dark, tough it is, but that the pattern was too small, and you can see the harsh reality intrude in the bottom-left corner.


