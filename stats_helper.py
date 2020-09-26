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
  subfolders = []
  dataset = np.array([])
  # subfolders under test/train folders
  # for trFolders in os.listdir(dir_name):
  #   subFoldersPath = os.path.join(dir_name, trFolders)
  #   # print(subFoldersPath)
  #   for i in os.listdir(subFoldersPath):
  #     subfolders.append(subFoldersPath + '/' + i)
  testfolder = os.path.join(dir_name, os.listdir(dir_name)[0])
  trainfolder = os.path.join(dir_name, os.listdir(dir_name)[1])
  print(trainfolder)
  for testSub in os.listdir(testfolder):
      subfolders.append(testfolder + '/' + testSub)
  for trainSub in os.listdir(trainfolder):
      subfolders.append(trainfolder + '/' + trainSub)
  print(subfolders)
  # # images under subfolders
  for folders in subfolders:
    pilImg = []
    for filename in os.listdir(folders):
      filename = folders + '/' + filename
      print(filename)
      pilImg = Image.open(filename).convert('L')
      pilImg = np.array(pilImg).astype('float32')
      pilImg /= 255
      pilImg = pilImg.flatten()
      dataset = np.append(dataset, pilImg)
  dataset = np.expand_dims(dataset, axis=1)
  scaler = StandardScaler().partial_fit(dataset)
  mean = scaler.mean_
  std = scaler.scale_
  print(f'mean: {mean}')
  print(f'std: {std}')
  # raise NotImplementedError('compute_mean_and_std not implemented')

  ############################################################################
  # Student code end
  ############################################################################
  return mean, std
