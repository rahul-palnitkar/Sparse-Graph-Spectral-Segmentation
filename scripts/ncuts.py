import numpy as np
from scipy.sparse.linalg import eigs
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
import seaborn as sns
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)


def ncuts(image, sig_I, sig_X):

    """
    Uses the Normalized Cuts method to segment an input greyscale image.
    image: a greyscale square image
    sig_I: a weight factor concerning similarity of pixel colors
    sig_X: a weight factor concerning distance between pixels
    return: (i_vec, labels), respectively the flattened image np.array
            and a flattened np.array of the segmented image
    """

    # flattened image, where columns are stacked (0,0), (1, 0),..., (4, 0), (0, 1),..., (4, 1), ... , (4, 4)
    i_vec = image.flatten('F')

    position_vec = np.zeros((i_vec.shape[0], 2))
    im_dim = image.shape[0]

    col_counter = 0
    for i in range(len(i_vec)):
        position_vec[i, 0] = i%im_dim
        position_vec[i, 1] = col_counter
        if i !=0 and i%im_dim == 0:
            col_counter += 1

    i_expand = np.expand_dims(i_vec, axis = 1)

    i_pdist = squareform(pdist(i_expand/(2 * (sig_I) ** 2), "sqeuclidean"))
    p_pdist = squareform(pdist(position_vec/(2 * (sig_X) ** 2) , "sqeuclidean"))

    
    
    W = np.exp(-(i_pdist + p_pdist)) + np.random.normal(0, 0.0001, size=i_pdist.shape)
    D = np.diag(np.matmul(W, np.ones(W.shape[1])))
    # L = D - W
    
    inv = np.diag(1/np.diag(D))

    eigenvalues, eigenvectors = eigs(np.matmul(inv, W), k=2)
    vec = np.expand_dims(eigenvectors[:, 1].astype(float), axis = 1)


    kmeans = KMeans(n_clusters=2).fit(vec)
    labels = kmeans.labels_.astype(bool)
    

    return i_vec, labels
    