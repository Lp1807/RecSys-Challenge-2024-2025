import implicit
from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
import scipy.sparse as sps
import numpy as np
class ImplicitALSRecommender(BaseMatrixFactorizationRecommender):
    """ImplicitALSRecommender recommender"""

    RECOMMENDER_NAME = "ImplicitALSRecommender"
    
    def linear_scaling_confidence(self, alpha=1.0):

        C = self.check_matrix(self.URM_train, format="csr", dtype = np.float32)
        C.data = 1.0 + alpha*C.data

        return C

    def _log_scaling_confidence(self, alpha=1.0, epsilon=1e-6):

        C = self.check_matrix(self.URM_train, format="csr", dtype = np.float32)
        C.data = 1.0 + self.alpha * np.log(1.0 + C.data / epsilon)

        return C
    
    def check_matrix(self, X, format='csc', dtype=np.float32):
        """
        This function takes a matrix as input and transforms it into the specified format.
        The matrix in input can be either sparse or ndarray.
        If the matrix in input has already the desired format, it is returned as-is
        the dtype parameter is always applied and the default is np.float32
        :param X:
        :param format:
        :param dtype:
        :return:
        """


        if format == 'csc' and not isinstance(X, sps.csc_matrix):
            return X.tocsc().astype(dtype)
        elif format == 'csr' and not isinstance(X, sps.csr_matrix):
            return X.tocsr().astype(dtype)
        elif format == 'coo' and not isinstance(X, sps.coo_matrix):
            return X.tocoo().astype(dtype)
        elif format == 'dok' and not isinstance(X, sps.dok_matrix):
            return X.todok().astype(dtype)
        elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
            return X.tobsr().astype(dtype)
        elif format == 'dia' and not isinstance(X, sps.dia_matrix):
            return X.todia().astype(dtype)
        elif format == 'lil' and not isinstance(X, sps.lil_matrix):
            return X.tolil().astype(dtype)

        elif format == 'npy':
            if sps.issparse(X):
                return X.toarray().astype(dtype)
            else:
                return np.array(X)

        elif isinstance(X, np.ndarray):
            X = sps.csr_matrix(X, dtype=dtype)
            X.eliminate_zeros()
            return self.check_matrix(X, format=format, dtype=dtype)
        else:
            return X.astype(dtype)

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