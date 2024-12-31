'''
@author: Luca Pagano: This class is an adaptation of the fit method present
in the example in 'Exercises/Practice extra 11 - Deep Learning Recommenders.ipynb'.
'''

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from tqdm import tqdm

class GraphConvolution(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "GraphConvolution"

    def __init__(self, URM_train, verbose=True):
        super(GraphConvolution, self).__init__(URM_train, verbose=verbose)

        self.R_tilde = None
        self.V = None
        self.D_I = None
        self.D_I_inv = None
        self.D_U_inv = None
        self.alpha = None


    def fit(self, alpha=1.0, num_factors=50, random_seed = None):
            self._print("Computing SVD decomposition of the normalized adjacency matrix...")

            self.alpha = alpha

            self.D_I = np.sqrt(np.array(self.URM_train.sum(axis = 0))).squeeze()
            self.D_I_inv = 1/(self.D_I + 1e-6)
            self.D_U_inv = 1/np.sqrt(np.array(self.URM_train.sum(axis = 1))).squeeze() + 1e-6

            self.D_I = sp.diags(self.D_I)
            self.D_I_inv = sp.diags(self.D_I_inv)
            self.D_U_inv = sp.diags(self.D_U_inv)

            self.R_tilde = self.D_U_inv.dot(self.URM_train).dot(self.D_I_inv) # <- Normalized URM

            _, _, self.V = randomized_svd(self.R_tilde,
                                        n_components = num_factors,
                                        random_state = random_seed) # <- Obtain V_K using truncated SVD

            self.D_I = sp.csr_matrix(self.D_I)
            self.D_I_inv = sp.csr_matrix(self.D_I_inv)
            
            print("Computing first term")
            first_term = self.R_tilde.T.dot(self.R_tilde)
            first_term = sp.csr_matrix(first_term)
            del self.R_tilde
            if self.verbose:
                print("D_I_inv shape:", self.D_I_inv.shape)
                print("V shape:", self.V.shape)
                print("D_I shape:", self.D_I.shape)

            # Had to transpose V to make the dot product work -> gpt agrees with me
            print("Computing second term: V^T * V")
            VTDotV = self.V.T.dot(self.V)
            VTDotV = sp.csr_matrix(VTDotV)
            
            print("Computing third term: alpha * D_I_inv * V^T * V")
            partial_res = self.alpha * self.D_I_inv.dot(VTDotV)
            partial_res = sp.csr_matrix(partial_res)
            del VTDotV, self.D_I_inv
            print("Computing fourth term: third_term * D_I")
            second_term = partial_res.dot(self.D_I)
            second_term = sp.csr_matrix(second_term)
            del partial_res, self.D_I
            self.W_sparse = first_term + second_term
            self.W_sparse = sp.csr_matrix(self.W_sparse)
            del first_term, second_term
            print("Computed Similarity matrix")