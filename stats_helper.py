import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> Tuple[np.ndarray, np.array]:
  '''
  Compute the mean and the standard deviation of the dataset.

  Note: convert the image in grayscale and then scale to [0,1] before computing
  mean and standard deviation

  Hints: use StandardScalar (check import statement)

  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''

  mean = None
  std = None

  ############################################################################
  # Student code begin
  ############################################################################
  files = glob.glob(os.path.join(dir_name, '*', '*', '*.jpg')) 
  scaler = StandardScaler() 
  for fileName in files: 
      with open(fileName, 'rb') as f: 
          img = np.asarray(Image.open(f).convert('L'), dtype='float32')
          img /= 255.0 
          scaler.partial_fit(img.reshape(-1, 1)) 
          mean = scaler.mean_ 
          std = scaler.scale_ 

  ############################################################################
  # Student code end
  ############################################################################
  return mean, std
