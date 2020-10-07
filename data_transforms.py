'''
Contains functions with different data transforms
'''

from typing import Tuple

import numpy as np
import torchvision.transforms as transforms


def get_fundamental_transforms(inp_size: Tuple[int, int],
                               pixel_mean: np.array,
                               pixel_std: np.array) -> transforms.Compose:
  '''
  Returns the core transforms needed to feed the images to our model
<<<<<<< HEAD
=======
  1. Resize the input image to the desired shape;
  2. Convert it to a tensor;
  3. Normalize them based on the computed mean and standard deviation.
>>>>>>> 2960f1699cce12147d2912bfc986493cde2075b9

  Args:
  - inp_size: tuple denoting the dimensions for input to the model
  - pixel_mean: the mean of the raw dataset [Shape=(1,)]
  - pixel_std: the standard deviation of the raw dataset [Shape=(1,)]
  Returns:
  - fundamental_transforms: transforms.Compose with the fundamental transforms
  '''
<<<<<<< HEAD

=======
 
>>>>>>> 2960f1699cce12147d2912bfc986493cde2075b9
  return transforms.Compose([
      ############################################################################
      # Student code begin
      ############################################################################
<<<<<<< HEAD
      transforms.Resize(inp_size),
=======
			transforms.Resize(inp_size),
>>>>>>> 2960f1699cce12147d2912bfc986493cde2075b9
			transforms.ToTensor(),
			transforms.Normalize(pixel_mean,pixel_std)
      ############################################################################
      # Student code end
      ############################################################################
  ])


def get_data_augmentation_transforms(inp_size: Tuple[int, int],
                                     pixel_mean: np.array,
                                     pixel_std: np.array) -> transforms.Compose:
  '''
<<<<<<< HEAD
  Returns the data augmentation + core transforms needed to be applied on the train set. Put data augmentation transforms before code transforms. 
=======
  Returns the data augmentation + core transforms needed to be applied on the train set. Put data augmentation transforms before core transforms. 
>>>>>>> 2960f1699cce12147d2912bfc986493cde2075b9

  Note: You can use transforms directly from torchvision.transforms

  Suggestions: Jittering, Flipping, Cropping, Rotating.

  Args:
  - inp_size: tuple denoting the dimensions for input to the model
  - pixel_mean: the mean of the raw dataset
  - pixel_std: the standard deviation of the raw dataset
  Returns:
  - aug_transforms: transforms.compose with all the transforms
  '''

  return transforms.Compose([
      ############################################################################
      # Student code begin
      ############################################################################

      ############################################################################
      # Student code end
      ############################################################################
  ])
