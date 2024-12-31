'''
@author Nicola Cesare
It is a wrapper of the implicit recommender (https://github.com/benfred/implicit).
@author: Luca Pagano: Added some modifications to make it work.
'''


import implicit
from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
import scipy.sparse as sps
import numpy as np
from Recommenders.Recommender_utils import check_matrix

class ImplicitALSRecommender(BaseMatrixFactorizationRecommender):
    """ImplicitALSRecommender recommender"""

    RECOMMENDER_NAME = "ImplicitALSRecommender"
    
    def linear_scaling_confidence(self, alpha=1.0):

        C = check_matrix(self.URM_train, format="csr", dtype = np.float32)
        C.data = 1.0 + alpha*C.data

        return C

    def _log_scaling_confidence(self, alpha=1.0, epsilon=1e-6):

        C = check_matrix(self.URM_train, format="csr", dtype = np.float32)
        C.data = 1.0 + self.alpha * np.log(1.0 + C.data / epsilon)

        return C

    def fit(self,
            factors=100,
            regularization=0.01,
            use_native=True, use_cg=True, use_gpu=False,
            iterations=15,
            calculate_training_loss=False, num_threads=0,
            confidence_scaling=None,
            icm_coeff = 1,
            **confidence_args
            ):
        self.rec = implicit.als.AlternatingLeastSquares(factors=factors, regularization=regularization,
                                                        use_native=use_native, use_cg=use_cg, use_gpu=use_gpu,
                                                        iterations=iterations,
                                                        calculate_training_loss=calculate_training_loss,
                                                        num_threads=num_threads,
                                                        random_state=42)
        tmp = self.linear_scaling_confidence(**confidence_args)

        self.rec.fit(tmp, show_progress=self.verbose)

        self.USER_factors = self.rec.user_factors
        self.ITEM_factors = self.rec.item_factors