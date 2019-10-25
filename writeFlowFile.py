"""
Write flow file
According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
Contact: dqsun@cs.brown.edu
Contact: schar@middlebury.edu
Updated to python3.7 etc. by Celyn Walters
Contact: c.walters@surrey.ac.uk

Original author: Johannes Oswald, Technical University Munich
Contact: johannes.oswald@tum.de

For more information, check http://vision.middlebury.edu/flow/
"""
from pathlib import Path
import numpy as np

TAG_STRING = b"PIEH"

# ==================================================================================================
def write(flow: np.array, path: Path):
	if (path.suffix != ".flo"):
		raise Exception(f"file extension is not `.flo`: {path}")

	height, width, nBands = flow.shape
	if (nBands != 2):
		raise ValueError(f"Number of bands: {nBands} != 2")

	u = flow[:, :, 0]
	v = flow[:, :, 1]
	if (u.shape != v.shape):
		raise ValueError(f"Flow shape mismatch: {u.shape} vs. {v.shape}")
	height, width = u.shape

	with open(path, "wb") as file:
		file.write(TAG_STRING)
		np.array(width).astype(np.int32).tofile(file)
		np.array(height).astype(np.int32).tofile(file)
		tmp = np.zeros((height, width*nBands))
		tmp[:,np.arange(width)*2] = u
		tmp[:,np.arange(width)*2 + 1] = v
		tmp.astype(np.float32).tofile(file)
