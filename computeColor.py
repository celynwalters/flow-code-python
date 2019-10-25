#!/usr/bin/env python3
"""
Compute coloured image to visualize optical flow file `.flo`
According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
Contact: dqsun@cs.brown.edu
Contact: schar@middlebury.edu
Updated to python3.7 etc. by Celyn Walters
Contact: c.walters@surrey.ac.uk

Original author: Johannes Oswald, Technical University Munich
Contact: johannes.oswald@tum.de

For more information, check http://vision.middlebury.edu/flow/
"""
import argparse
from pathlib import Path
import cv2
import sys
import numpy as np

import readFlowFile

# ==================================================================================================
def makeColorwheel() -> np.array:
	"""
	Color encoding scheme adapted from the color circle idea described at http://members.shaw.ca/quadibloc/other/colint.htm

	Returns:
		np.array: Colorwheel
	"""
	RY = 15
	YG = 6
	GC = 4
	CB = 11
	BM = 13
	MR = 6

	ncols = RY + YG + GC + CB + BM + MR
	colorwheel = np.zeros([ncols, 3]) # R, G, B

	col = 0
	# RY
	colorwheel[0:RY, 0] = 255
	colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY, 1) / RY)
	col += RY

	# YG
	colorwheel[col:YG + col, 0] = 255 - np.floor(255 * np.arange(0, YG, 1) / YG)
	colorwheel[col:YG + col, 1] = 255
	col += YG

	# GC
	colorwheel[col:GC + col, 1] = 255
	colorwheel[col:GC + col, 2] = np.floor(255 * np.arange(0, GC, 1) / GC)
	col += GC

	# CB
	colorwheel[col:CB + col, 1] = 255 - np.floor(255*np.arange(0, CB, 1) / CB)
	colorwheel[col:CB + col, 2] = 255
	col += CB

	# BM
	colorwheel[col:BM + col, 2] = 255
	colorwheel[col:BM + col, 0] = np.floor(255 * np.arange(0, BM, 1) / BM)
	col += BM

	# MR
	colorwheel[col:MR + col, 2] = 255 - np.floor(255 * np.arange(0, MR, 1) / MR)
	colorwheel[col:MR + col, 0] = 255

	return colorwheel


# ==================================================================================================
def computeColor(u: float, v: float) -> np.array:
	"""
	Get the colour in the wheel at the specified coordinates.

	Args:
		u (float): X coordinate
		v (float): Y coordinate

	Returns:
		np.array:
	"""
	colorwheel = makeColorwheel()
	nan_u = np.isnan(u)
	nan_v = np.isnan(v)
	nan_u = np.where(nan_u)
	nan_v = np.where(nan_v)

	u[nan_u] = 0
	u[nan_v] = 0
	v[nan_u] = 0
	v[nan_v] = 0

	ncols = colorwheel.shape[0]
	radius = np.sqrt(u**2 + v**2)
	a = np.arctan2(-v, -u) / np.pi
	fk = (a + 1) / 2 * (ncols - 1) # -1~1 maped to 1~ncols
	k0 = fk.astype(np.uint8)	 # 1, 2, ..., ncols
	k1 = k0 + 1
	k1[k1 == ncols] = 0
	f = fk - k0

	img = np.empty([k1.shape[0], k1.shape[1], 3])
	ncolors = colorwheel.shape[1]
	for i in range(ncolors):
		tmp = colorwheel[:, i]
		col0 = tmp[k0] / 255
		col1 = tmp[k1] / 255
		col = ((1 - f) * col0) + (f * col1)
		idx = radius <= 1
		col[idx] = 1 - radius[idx] * (1 - col[idx]) # Increase saturation with radius
		col[~idx] *= 0.75 # out of range
		img[:, :, 2 - i] = np.floor(255 * col).astype(np.uint8)

	return img.astype(np.uint8)


# ==================================================================================================
def computeImg(flow) -> np.array:
	"""
	Compute the colour-coded flow image.

	Args:
		flow (np.array): Flow field

	Returns:
		np.array: Colour-coded image
	"""
	eps = sys.float_info.epsilon
	UNKNOWN_FLOW_THRESH = 1e9
	UNKNOWN_FLOW = 1e10

	u = flow[:, :, 0]
	v = flow[:, :, 1]

	maxu = -999
	maxv = -999

	minu = 999
	minv = 999

	maxrad = -1
	# Fix unknown flow
	greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
	greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
	u[greater_u] = 0
	u[greater_v] = 0
	v[greater_u] = 0
	v[greater_v] = 0

	maxu = max([maxu, np.amax(u)])
	minu = min([minu, np.amin(u)])

	maxv = max([maxv, np.amax(v)])
	minv = min([minv, np.amin(v)])
	rad = np.sqrt(np.multiply(u, u) + np.multiply(v, v))
	maxrad = max([maxrad, np.amax(rad)])
	print(f"max flow:")
	print(f"  {maxrad:.4f}")
	print(f"flow range:")
	print(f"  u = {minu:.3f} .. {maxu:.3f}")
	print(f"  v = {minv:.3f} .. {maxv:.3f}")

	u = u / (maxrad + eps)
	v = v / (maxrad + eps)
	img = computeColor(u, v)

	return img


# ==================================================================================================
if (__name__ == "__main__"):
	parser = argparse.ArgumentParser()
	parser.add_argument("flow_file", type=str, default="colorTest.flo", help="Flow file")
	parser.add_argument("--write", action="store_true", help="Write flow as PNG")
	args = parser.parse_args()
	args.flow_file = Path(args.flow_file)
	flow = readFlowFile.read(args.flow_file)
	img = computeImg(flow)
	cv2.imshow(str(args.flow_file), img)
	k = cv2.waitKey()
	if parser.parse_args().write:
		cv2.imwrite(str(args.flow_file.with_suffix(".png")), img)
