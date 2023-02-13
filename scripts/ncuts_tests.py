""" Test ncuts segmentation method on synthetic noisy images"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jaccard
from sklearn.cluster import KMeans
import time
import pandas as pd
from PIL import Image


from ncuts import ncuts
from utility import noisy

import warnings
warnings.simplefilter("ignore", np.ComplexWarning)


""" 
INDICATE WHAT TYPE OF TEST YOU'D LIKE TO RUN:
- Preset Gaussian-Noisy Synthetic Image Tests: Set TEST_TYPE = "PRESET GAUSS"
- Preset Salt-and-Pepper Noisy Synthetic Image Tests: Set TEST_TYPE = "PRESET SP"

"""


TEST_TYPE  = "PRESET SP"

"""IF CUSTOM, PLEASE INDICATE IMAGE PATH"""

IMAGE_PATH = "figures/knot.jpg"

def main():
    if TEST_TYPE == "PRESET GAUSS":
        return synthetic_gauss_tests()
    elif TEST_TYPE == "PRESET SP":
        return synthetic_sp_tests()
    else:
        raise Exception(f"{TEST_TYPE} is not a valid test type!")


######################################################
#               TEST SUITE FUNCTIONS                 #
######################################################


def synthetic_gauss_tests():
    """ TABLE TESTS """
    size = 100

    ncut_times = []
    for gt in [4, 5, 6]:
        gt_im = (Image.open(f'figures/gt_{gt}.png').convert('L')).resize((size, size))
        ground_truth = np.asarray(gt_im).astype(bool).astype(int)
        plt.imsave(f'preset-synthetic-tests/existing-methods/ncuts/gt_{gt}_100x100.png', ground_truth, cmap='gray')
        plt.clf()
        flat_gt = ground_truth.flatten('F')
        
        nlist = [0.5, 1.5]
        for noise in range(len(nlist)):
            noise_matrix = np.random.normal(0, nlist[noise], ground_truth.shape)


            noisy_image = (ground_truth + noise_matrix)
            uvs = np.unique(noisy_image)

            img_matrix_c = (((noisy_image - uvs[0])/(uvs[-1] - uvs[0])) * 255).astype(int)

            plt.imshow(img_matrix_c, cmap = 'gray')
            plt.imsave(f'preset-synthetic-tests/existing-methods/ncuts/gauss/gt{gt}_gauss{nlist[noise]}_100x100.png', img_matrix_c, cmap = 'gray')
            plt.clf()



            list_I = [0.1, 79]
            list_X = np.linspace(0.1, size, 10)

            for i in range(len(list_I)):
                for x in range(len(list_X)):
                    start_ncuts_segment = time.time()
                    ncuts_segment = ncuts(img_matrix_c, list_I[i], list_X[x])[1]
                    end_ncuts_segment = time.time() - start_ncuts_segment
                    seg_image_ncuts = ncuts_segment.reshape((img_matrix_c.shape[0], img_matrix_c.shape[1]), order="F")
                    plt.imsave(f"preset-synthetic-tests/existing-methods/ncuts/gauss/gt_{gt}/ncuts_I{list_I[i]}_X{list_X[x]}_noise{noise}.png" , seg_image_ncuts, cmap = "gray")
                    plt.clf()
                    ncuts_jacc = min(jaccard(flat_gt, ncuts_segment), jaccard(flat_gt,  np.logical_not(ncuts_segment)))
                    ncut_times.append([gt, nlist[noise], list_I[i], list_X[x], ncuts_jacc, end_ncuts_segment])
            

    
    ncuts_df = pd.DataFrame (ncut_times, columns = ["gt", "noise", "sigma_I", "sigma_X", "jaccard_dissimilarity", "time"])
    ncuts_df.to_csv('preset-synthetic-tests/existing-methods/ncuts/gauss/ncuts_times.csv')




def synthetic_sp_tests():
    """ TABLE TESTS """
    size = 100

    ncut_times = []
    for gt in [4, 5, 6]:
        gt_im = (Image.open(f'figures/gt_{gt}.png').convert('L')).resize((size, size))
        ground_truth = np.asarray(gt_im).astype(bool).astype(int)
        plt.imsave(f'preset-synthetic-tests/existing-methods/ncuts/gt_{gt}_100x100.png', ground_truth, cmap = 'gray')
        plt.clf()
        flat_gt = ground_truth.flatten('F')
        
        nlist = [0.3, 0.7]
        for noise in range(len(nlist)):
            
            #noisy_image = ground_truth + random_noise(ground_truth, mode='s&p',amount=nlist[noise])

            noisy_image = noisy("s&p", ground_truth, nlist[noise])
            uvs = np.unique(noisy_image)

            img_matrix_c = (((noisy_image - uvs[0])/(uvs[-1] - uvs[0])) * 255).astype(int)

            plt.imsave(f'preset-synthetic-tests/existing-methods/ncuts/sp/gt{gt}_saltpepper{noise}_100x100.png', img_matrix_c, cmap = 'gray')
            plt.clf()



            list_I = [0.1, 79]
            list_X = np.linspace(0.1, size, 10)

            for i in range(len(list_I)):
                for x in range(len(list_X)):
                    start_ncuts_segment = time.time()
                    ncuts_segment = ncuts(img_matrix_c, list_I[i], list_X[x])[1]
                    end_ncuts_segment = time.time() - start_ncuts_segment
                    seg_image_ncuts = ncuts_segment.reshape((img_matrix_c.shape[0], img_matrix_c.shape[1]), order="F")
                    plt.imsave(f"preset-synthetic-tests/existing-methods/ncuts/sp/gt_{gt}/ncuts_I{list_I[i]}_X{list_X[x]}_noise{noise}.png" , seg_image_ncuts, cmap = "gray")
                    plt.clf()
                    ncuts_jacc = min(jaccard(flat_gt, ncuts_segment), jaccard(flat_gt,  np.logical_not(ncuts_segment)))
                    ncut_times.append([gt, nlist[noise], list_I[i], list_X[x], ncuts_jacc, end_ncuts_segment])
            

    
    ncuts_df = pd.DataFrame (ncut_times, columns = ["gt", "noise", "sigma_I", "sigma_X", "jaccard_dissimilarity", "time"])
    ncuts_df.to_csv('preset-synthetic-tests/existing-methods/ncuts/sp/ncuts_times.csv')




if __name__ == "__main__":
        main()

