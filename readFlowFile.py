"""
Compute coloured image to visualize optical flow file `.flo`
According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
Contact: dqsun@cs.brown.edu
Updated to python3.7 etc. by Celyn Walters
Contact: c.walters@surrey.ac.uk
Contact: schar@middlebury.edu

Original author: Johannes Oswald, Technical University Munich
Contact: johannes.oswald@tum.de

For more information, check http://vision.middlebury.edu/flow/
"""
from pathlib import Path
import numpy as np

TAG_FLOAT = 202021.25

# ==================================================================================================
def read(path: Path):
	if not path.exists():
		raise FileNotFoundError(f"File does not exist: {path}")
	if not path.is_file():
		raise Exception(f"Exists but is not a file: {path}")
	if (path.suffix != ".flo"):
		raise Exception(f"file extension is not `.flo`: {path}")

	with open(path, "rb") as file:
		floNumber = np.fromfile(file, np.float32, count=1)[0]
		if (floNumber != TAG_FLOAT):
			raise ValueError(f"Flow number {floNumber} incorrect. Invalid .flo file")
		w = np.fromfile(file, np.int32, count=1)
		h = np.fromfile(file, np.int32, count=1)
		# data = np.fromfile(f, np.float32, count=2*w*h)
		data = np.fromfile(file, np.float32, count=(2 * w[0] * h[0]))
		flow = np.resize(data, (int(h), int(w), 2)) # Reshape data into 3D array (columns, rows, bands)

	return flow
