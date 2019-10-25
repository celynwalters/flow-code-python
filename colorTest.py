#!/usr/bin/env python3
"""
Creates a test image showing the color encoding scheme
According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
Contact: dqsun@cs.brown.edu
Contact: schar@middlebury.edu
Updated to python3.7 etc. by Celyn Walters
Contact: c.walters@surrey.ac.uk

Original author: Johannes Oswald, Technical University Munich
Contact: johannes.oswald@tum.de

For more information, check http://vision.middlebury.edu/flow/
"""
import numpy as np
import cv2
import math
from pathlib import Path

import computeColor
import writeFlowFile
import readFlowFile

# ==================================================================================================
if (__name__ == "__main__"):
	truerange = 1
	height = 151
	width = 151
	range_f = truerange * 1.04

	s2 = int(round(height / 2))
	x, y = np.meshgrid(np.arange(1, height + 1, 1), np.arange(1, width + 1, 1))

	u = x * range_f / s2 - range_f
	v = y * range_f / s2 - range_f

	img = computeColor.computeColor(u / truerange, v / truerange)

	img[s2, :, :] = 0
	img[:, s2, :] = 0

	cv2.imshow("Test color pattern", img)
	k = cv2.waitKey()

	F = np.stack((u, v), axis=2)
	writeFlowFile.write(F, Path("colorTest.flo"))

	flow = readFlowFile.read(Path("colorTest.flo"))

	u = flow[:, :, 0]
	v = flow[:, :, 1]

	img = computeColor.computeColor(u / truerange, v / truerange)

	img[s2, :, :] = 0
	img[:, s2, :] = 0

	cv2.imshow("Saved and reloaded test color pattern", img)
	k = cv2.waitKey()

	# Color encoding scheme for optical flow
	img = computeColor.computeColor(u / range_f / math.sqrt(2), v / range_f / math.sqrt(2))

	cv2.imshow("Optical flow color encoding scheme", img)
	cv2.imwrite("colorTest.png", img)
	k = cv2.waitKey()
