""" Test proposed segmentation method on synthetic noisy images, preset color images, and custom images"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack, eye, spdiags, identity
from scipy.sparse.linalg import eigs, eigsh
from scipy.spatial.distance import pdist, squareform, jaccard, cdist
from sklearn.cluster import KMeans
import time
import pandas as pd
from PIL import Image


from extranodes import segment
from utility import noisy

import warnings
warnings.simplefilter("ignore", np.ComplexWarning)



""" 
INDICATE WHAT TYPE OF TEST YOU'D LIKE TO RUN:
- Custom Color Image: Set TEST_TYPE = "COLOR"
- Custom Synthetic Image: Set TEST_TYPE = "SYNTHETIC"
"""

TEST_TYPE  = "SYNTHETIC"

"""FOR COLOR: PLEASE INDICATE IMAGE PATH"""

IMAGE_PATH = "figures/knot.jpg"

"""IF SYNTHETIC IMAGE PLEASE CHOOSE GROUND TRUTH (1-6) AND NOISE TYPE (SP or GAUSS)"""
GT = 4
NOISE = "GAUSS"
NOISE_LEVEL = 0.5


"""SPECIFY MU AND SIGMA"""
SIGMA_C = 0
MU = 3

def main():
    if TEST_TYPE == "COLOR":
        return  color_seg(IMAGE_PATH, SIGMA_C, MU)
    elif TEST_TYPE == "SYNTHETIC":
        return synthetic_seg(GT, NOISE, NOISE_LEVEL, SIGMA_C, MU, size = 100)
    else:
        raise Exception(f"{TEST_TYPE} is not a valid test type!")




######################################################
#               TEST SUITE FUNCTIONS                 #
######################################################


def color_seg(img_path, sigma_C, mu):
    time_array = []
    img_rgb = (Image.open(img_path))
    img_rgb = np.asarray(img_rgb)

    (h, w, c) = img_rgb.shape
    flat_img = img_rgb.reshape(h * w, c)
    start_cluster = time.time()
    kmeans_model = KMeans(n_clusters=256).fit(flat_img)
    cluster_labels = kmeans_model.fit_predict(flat_img)
    img_labels = cluster_labels.reshape(h, w)
    print(f"Clustering Colors in time {time.time() - start_cluster}")

    real_img_array  = np.reshape(img_labels, (img_rgb.shape[0], img_rgb.shape[1]), order = "C").astype("uint8")

    
    start_seg = time.time()
    img_segment = segment(real_img_array, sigma_C, mu)
    end_seg = time.time()
    seg_time = end_seg - start_seg
    print(f"segmentation finished in time {seg_time} seconds")

    seg_image_color = (img_segment[0]).reshape((real_img_array.shape[0], real_img_array.shape[1]), order="C")
    plt.imshow(seg_image_color, cmap = "gray")
    plt.axis('off')
    plt.title(f"σ = {SIGMA_C}, µ = {MU}")
    plt.show()
    

def synthetic_seg(gt, noise, noise_level, sigma_C, mu, size):
    gt_im = (Image.open(f'figures/gt_{gt}.png').convert('L')).resize((size, size))
    ground_truth = np.asarray(gt_im).astype(bool).astype(int)
    flat_gt = ground_truth.flatten('C')
    if noise == "SP":
        noisy_image = noisy("s&p", ground_truth, noise_level)
        uvs = np.unique(noisy_image)
        img_matrix_c = (((noisy_image - uvs[0])/(uvs[-1] - uvs[0])) * 255).astype(int)
    elif noise == "GAUSS":
        noise_matrix = np.random.normal(0, noise_level, ground_truth.shape)
        noisy_image = (ground_truth + noise_matrix)
        uvs = np.unique(noisy_image)
        img_matrix_c = (((noisy_image - uvs[0])/(uvs[-1] - uvs[0])) * 255).astype(int)
    else:
        raise Exception(f"{NOISE} is not a valid noise type!")
    
    start_seg = time.time()
    img_segment = segment(img_matrix_c, sigma_C, mu)
    end_seg = time.time()
    seg_time = end_seg - start_seg
    print(f"segmentation finished in time {seg_time} seconds")
    jac = min(jaccard(flat_gt, img_segment[2]), jaccard(flat_gt,  np.logical_not(img_segment[2])))

    seg_image_synth = (img_segment[0]).reshape((img_matrix_c.shape[0], img_matrix_c.shape[1]), order="C")
    plt.imshow(seg_image_synth, cmap = "gray")
    plt.axis('off')
    plt.title(f"σ = {SIGMA_C}, µ = {MU}, {noise} at {noise_level}, JAC = {jac}")
    plt.show()






if __name__ == "__main__" :
    main()