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
from skimage.util import random_noise


from extranodes import segment
from utility import noisy

import warnings
warnings.simplefilter("ignore", np.ComplexWarning)


""" 
INDICATE WHAT TYPE OF TEST YOU'D LIKE TO RUN:
- Custom Image: Set TEST_TYPE = "CUSTOM"
- Preset Gaussian-Noisy Synthetic Image Tests: Set TEST_TYPE = "PRESET GAUSS"
- Preset Salt-and-Pepper Noisy Synthetic Image Tests: Set TEST_TYPE = "PRESET SP"
- Preset Color segmentation tests: Set TEST_TYPE = "PRESET COLOR"

"""


TEST_TYPE  = "PRESET GAUSS"

"""IF CUSTOM, PLEASE INDICATE IMAGE PATH"""

IMAGE_PATH = "figures/knot.jpg"

def main():
    if TEST_TYPE == "CUSTOM":
        return  custom_seg(IMAGE_PATH)
    elif TEST_TYPE == "PRESET GAUSS":
        return synthetic_gauss_tests()
    elif TEST_TYPE == "PRESET SP":
        return synthetic_sp_tests()
    elif TEST_TYPE == "PRESET COlOR":
        return color_tests()
    else:
        raise Exception(f"{TEST_TYPE} is not a valid test type!")




######################################################
#               TEST SUITE FUNCTIONS                 #
######################################################


def custom_seg(img_path):
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

    
    sigma_C = 0
    list_L = np.append(np.linspace(0, 100, 25), [150, 200, 300, 400, 500])
    for l in list_L:
        start_seg = time.time()
        img_segment = segment(real_img_array, sigma_C, l)
        end_seg = time.time()
        seg_time = end_seg - start_seg
        print(f"segmentation finished in time {seg_time} seconds")

        seg_image_color = (img_segment[0]).reshape((real_img_array.shape[0], real_img_array.shape[1]), order="C")
        plt.imsave(f"custom-tests/sigma0_mu{l}_segment.png", seg_image_color, cmap = "gray")
        plt.clf()
        inverse_seg_image = np.logical_not((img_segment[0]).reshape((real_img_array.shape[0], real_img_array.shape[1]), order="C"))
        plt.imsave(f"custom-tests/sigma0_mu{l}_inverse.png", inverse_seg_image, cmap = "gray")
        plt.clf()
        seg_eigen =img_segment[3]
        plt.imsave(f"custom-tests/eigenvector/sigma0_mu{l}_eigenvector.png", seg_eigen, cmap = "gray")
        plt.clf()
        inv_eigen = -img_segment[3]
        plt.imsave(f"custom-tests/eigenvector/sigma0_mu{l}_eigenvector_inverse.png", inv_eigen, cmap = "gray")
        plt.clf()

        time_array.append([sigma_C, l, seg_time])

    mu_df = pd.DataFrame (time_array, columns = ["sigma_c", "mu", "time"])
    mu_df.to_csv(f'custom-tests/mu_color_times.csv')




def synthetic_gauss_tests():
    size = 100

    color_times = []
    ncut_times = []
    for gt in [4, 5, 6]:
        gt_im = (Image.open(f'figures/gt_{gt}.png').convert('L')).resize((size, size))
        ground_truth = np.asarray(gt_im).astype(bool).astype(int)
        plt.imsave(f'preset-synthetic-tests/gt_{gt}_100x100.png', ground_truth, cmap='gray')
        plt.clf()
        flat_gt = ground_truth.flatten('C')
        
        nlist = [0.5, 1.5]
        for noise in range(len(nlist)):
            noise_matrix = np.random.normal(0, nlist[noise], ground_truth.shape)


            noisy_image = (ground_truth + noise_matrix)
            uvs = np.unique(noisy_image)

            img_matrix_c = (((noisy_image - uvs[0])/(uvs[-1] - uvs[0])) * 255).astype(int)

            plt.imshow(img_matrix_c, cmap = 'gray')
            plt.imsave(f'preset-synthetic-tests/gauss/gt{gt}_gauss{nlist[noise]}_100x100.png', img_matrix_c, cmap = 'gray')
            plt.clf()



            list_C = [0, 3, 5, 10]
            list_L = np.linspace(0, 10, 10)

            for c in range(len(list_C)):
                for l in range(len(list_L)):
                    start_color_segment = time.time()
                    color_segment = segment(img_matrix_c, list_C[c], list_L[l])
                    end_color_segment = time.time() - start_color_segment
                    plt.imsave(f"preset-synthetic-tests/gauss/gt_{gt}/color_C{list_C[c]}_M{list_L[l]}_noise{nlist[noise]}.png" , color_segment[0], cmap = "gray")
                    plt.clf()
                    color_jacc = min(jaccard(flat_gt, color_segment[2]), jaccard(flat_gt,  np.logical_not(color_segment[2])))
                    color_times.append([gt, nlist[noise], list_C[c], list_L[l], color_jacc, end_color_segment])

    color_df = pd.DataFrame (color_times, columns = ["gt", "noise", "sigma_c", "mu", "jaccard_dissimilarity", "time"])
    color_df.to_csv('preset-synthetic-tests/gauss/extranodes_gauss_times.csv')



def synthetic_sp_tests():
    size = 100
    color_times = []
    for gt in [4, 5, 6]:
        gt_im = (Image.open(f'figures/gt_{gt}.png').convert('L')).resize((size, size))
        ground_truth = np.asarray(gt_im).astype(bool).astype(int)
        plt.imsave(f'preset-synthetic-tests/gt_{gt}_100x100.png', ground_truth, cmap = 'gray')
        plt.clf()
        flat_gt = ground_truth.flatten('C')
        
        nlist = [0.3, 0.7]
        for noise in range(len(nlist)):
            

            noisy_image = noisy("s&p", ground_truth, nlist[noise])
            uvs = np.unique(noisy_image)

            img_matrix_c = (((noisy_image - uvs[0])/(uvs[-1] - uvs[0])) * 255).astype(int)

            plt.imsave(f'preset-synthetic-tests/sp/gt{gt}_saltpepper{noise}_100x100.png', img_matrix_c, cmap = 'gray')
            plt.clf()



            list_C = np.linspace(0, 10, 11)
            list_L = np.linspace(0, 10, 11)

            for c in range(len(list_C)):
                for l in range(len(list_L)):
                    start_color_segment = time.time()
                    color_segment = segment(img_matrix_c, list_C[c], list_L[l])
                    end_color_segment = time.time() - start_color_segment
                    plt.imsave(f"preset-synthetic-tests/sp/gt_{gt}/color_C{list_C[c]}_M{list_L[l]}_sp{nlist[noise]}.png" , color_segment[0], cmap = "gray")
                    plt.clf()
                    color_jacc = min(jaccard(flat_gt, color_segment[2]), jaccard(flat_gt,  np.logical_not(color_segment[2])))
                    color_times.append([gt, nlist[noise], list_C[c], list_L[l], color_jacc, end_color_segment])
                    

    color_df = pd.DataFrame (color_times, columns = ["gt", "noise", "sigma_c", "mu", "jaccard_dissimilarity", "time"])
    color_df.to_csv('preset-synthetic-tests/sp/extranodes_sp_times.csv')


def color_tests():
    im_list = ["knot.jpg" , "banana.jpg", "banana3.jpg", "GT07.png"]
    time_array = []
    for base in im_list:

        img_rgb = (Image.open(f'figures/{base}'))
        img_rgb = np.asarray(img_rgb)

        (h, w, c) = img_rgb.shape
        flat_img = img_rgb.reshape(h * w, c)
        start_cluster = time.time()
        kmeans_model = KMeans(n_clusters=256).fit(flat_img)
        cluster_labels = kmeans_model.fit_predict(flat_img)
        img_labels = cluster_labels.reshape(h, w)
        print(f"Clustering Colors in time {time.time() - start_cluster}")

        real_img_array  = np.reshape(img_labels, (img_rgb.shape[0], img_rgb.shape[1]), order = "C").astype("uint8")

        
        sigma_C = 0
        list_L = np.append(np.linspace(0, 100, 25), [150, 200, 300, 400, 500])
        for l in list_L:
            start_seg = time.time()
            img_segment = segment(real_img_array, sigma_C, l)
            end_seg = time.time()
            seg_time = end_seg - start_seg




            seg_image_color = (img_segment[0]).reshape((real_img_array.shape[0], real_img_array.shape[1]), order="C")
            plt.imsave(f"mu-tests/{base[:-4]}/sigma0_mu{l}_{base[:-4]}_segment.png", seg_image_color, cmap = "gray")
            plt.clf()
            inverse_seg_image = np.logical_not((img_segment[0]).reshape((real_img_array.shape[0], real_img_array.shape[1]), order="C"))
            plt.imsave(f"mu-tests/{base[:-4]}/sigma0_mu{l}_{base[:-4]}_inverse.png", inverse_seg_image, cmap = "gray")
            plt.clf()
            seg_eigen =img_segment[3]
            plt.imsave(f"mu-tests/{base[:-4]}/sigma0_mu{l}_{base[:-4]}_eigenvector.png", seg_eigen, cmap = "gray")
            plt.clf()
            inv_eigen = -img_segment[3]
            plt.imsave(f"mu-tests/{base[:-4]}/sigma0_mu{l}_{base[:-4]}_eigenvector_inverse.png", inv_eigen, cmap = "gray")
            plt.clf()

            time_array.append([base[:-4], sigma_C, l, seg_time])

    mu_df = pd.DataFrame (time_array, columns = ["image",  "sigma_c", "mu", "time"])
    mu_df.to_csv(f'mu-tests/mu_color_times.csv')



if __name__ == "__main__" :
    main()