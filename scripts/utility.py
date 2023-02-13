
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack, eye, spdiags, identity
from scipy.sparse.linalg import eigs, eigsh
from scipy.spatial.distance import pdist, squareform, jaccard, cdist
from sklearn.cluster import KMeans
import seaborn as sns
import time
import cv2
import pandas as pd
from PIL import Image
from skimage.util import random_noise
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.simplefilter("ignore", FutureWarning)





def noisy(noise_typ,image, noise):
    if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
    elif noise_typ == "s&p":
      row,col= image.shape
      s_vs_p = 0.5
      amount = noise
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
    elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
    elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy