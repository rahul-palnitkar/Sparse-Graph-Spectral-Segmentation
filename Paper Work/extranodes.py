
import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack, eye, spdiags
from scipy.sparse.linalg import eigs, eigsh
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import time
import warnings

warnings.simplefilter("ignore", np.ComplexWarning)


def make_graph_position(img, mu):
    """ Creates G_grid """
    (r, c) = img.shape[0], img.shape[1]
    num_pix = img.shape[0]*img.shape[1]
    (ny, nx) = np.meshgrid(np.arange(0, r), np.arange(0, c), indexing='ij')
    ct = nx + ny * c
    le = nx + ny * c + 1;
    so = nx + (ny + 1) * c;

    edges_i = np.append(np.ndarray.flatten(ct[:, :-1], 'F'), np.ndarray.flatten(ct[:-1, :], 'F'))
    edges_j = np.append(np.ndarray.flatten(le[:, :-1], 'F'), np.ndarray.flatten(so[:-1, :], 'F'))
    data = np.ones(edges_i.shape)

    M = csr_matrix((data, (edges_i, edges_j)), shape=(num_pix, num_pix))
    M = M + M.T
    return M


def eig_dec_v1(W_color):
    """Eigenvector solver (unused)"""
    n = W_color.shape[0]
    D = spdiags(np.squeeze(W_color.sum(axis=1)), 0, n, n)
    D_inv_sqrt = spdiags(np.sqrt(1 / np.squeeze(W_color.sum(axis=1))), 0, n, n)
    D_sqrt = spdiags(np.sqrt(np.squeeze(W_color.sum(axis=1))), 0, n, n)

    M = D_inv_sqrt.dot(W_color).dot(D_inv_sqrt)

    eigenvalues, eigenvectors = eigsh(M, k=2, which='LM')
    eigenvectors = D_sqrt.dot(eigenvectors)
    return eigenvectors


def eig_dec_v2(W_color):
    """Compute eigenvectors of Laplacian"""
    n = W_color.shape[0]
    D = spdiags(np.squeeze(W_color.sum(axis=1)), 0, n, n)
    D_inv = spdiags(1 / np.squeeze(W_color.sum(axis=1)), 0, n, n)

    M = D_inv.dot(W_color)
    eigenvalues, eigenvectors = eigs(M, k=2, which="LM")
    return eigenvectors


def remove_colors(img):
    """Utility function to speed up color segmentations"""
    vec_img = np.squeeze(img.reshape(-1, 1))
    shape_img = img.shape
    colors_img = np.unique(vec_img.reshape(-1, 1))

    color_indices = np.zeros(max(colors_img) + 1).astype(int)
    c = 0
    for color in colors_img:
        color_indices[color] = int(c)
        c += 1

    new_vec_img = color_indices[np.ix_(vec_img)]
    new_img = new_vec_img.reshape(shape_img)
    return new_img

def segment(img, sig_I, mu):
    """
    Given image, sigma and mu, computes segmentation
    returns: a tuple consisting of the segmented image, 
    the original flattened image vector, the flattened segmentation, and
    the eigenvector of the image.
    """
    num_pix = img.shape[0] * img.shape[1]
    G = make_graph_position(img, mu)

    if (img.shape[0], img.shape[1]) == img.shape:
        i_vec = img.flatten('C')
        i_vec = i_vec.reshape(-1, 1)
        colors = (np.unique(i_vec, axis=0)).reshape(-1, 1)

    else:
        i_vec = np.transpose(img, (1, 0, 2)).reshape(num_pix, img.shape[2])
        colors = np.unique(i_vec, axis=0)

    C = np.zeros((i_vec.shape[0], colors.shape[0]))

    num_colors = colors.shape[0]
    i_expand = i_vec
    c_expand = colors
    if sig_I != 0:
        C = csr_matrix(
            np.around(np.exp(-cdist(i_expand, c_expand, metric="sqeuclidean") / (2 * sig_I ** 2)), decimals=10))

    else:

        start_C_0= time.time()
        C_temp = np.exp(-cdist(i_expand, c_expand, metric="sqeuclidean"))
        C = csr_matrix(C_temp == 1)

        I = np.eye(num_colors)

        # rc_image_flat = remove_colors(img).flatten("C")
        # rc_image_flat = img.flatten("C")
        # C = I[np.ix_(rc_image_flat)]

        #print("Computing C:",time.time() - start_C_0)

    I_c = eye(num_colors)
    W_left = vstack((mu * G, np.transpose(C)))
    W_right = vstack((C, I_c))
    W_color = hstack((W_left, W_right))

    start_t = time.time()
    eigenvectors = eig_dec_v2(W_color)

    vec = np.expand_dims(eigenvectors[:, 1].astype(float), axis=1)  # eigenvector order is diferent for eigsh
    kmeans = KMeans(n_clusters=2).fit(vec)
    labels = kmeans.labels_.astype(bool)[0:i_vec.shape[0]]
    eigen = vec[0:i_vec.shape[0]]

    seg = labels.reshape((img.shape[0], img.shape[1]), order="C")
    evec_img = eigen.reshape((img.shape[0], img.shape[1]), order="C")
    return seg, i_vec, labels, evec_img