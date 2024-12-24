#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Massimo Quadrana, Cesare Bernardis
"""


import numpy as np
import scipy.sparse as sps
from Recommenders.Recommender_utils import check_matrix
from sklearn.linear_model import ElasticNet
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Similarity.Compute_Similarity_Python import Incremental_Similarity_Builder
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
import time, sys
from tqdm import tqdm
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from concurrent.futures import ProcessPoolExecutor

# os.environ["PYTHONWARNINGS"] = ('ignore::exceptions.ConvergenceWarning:sklearn.linear_model')
# os.environ["PYTHONWARNINGS"] = ('ignore:Objective did not converge:ConvergenceWarning:')

class SLIMElasticNetRecommender(BaseItemSimilarityMatrixRecommender):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
          make use of half the cores available

    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.

        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    RECOMMENDER_NAME = "SLIMElasticNetRecommender"

    def __init__(self, URM_train, verbose = True):
        super(SLIMElasticNetRecommender, self).__init__(URM_train, verbose = verbose)

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, l1_ratio=0.1, alpha = 1.0, positive_only=True, topK = 100):

        assert l1_ratio>= 0 and l1_ratio<=1, "{}: l1_ratio must be between 0 and 1, provided value was {}".format(self.RECOMMENDER_NAME, l1_ratio)

        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.topK = topK


        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=alpha,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-4)

        URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)

        n_items = URM_train.shape[1]

        similarity_builder = Incremental_Similarity_Builder(self.n_items, initial_data_block=self.n_items*self.topK, dtype = np.float32)

        start_time = time.time()
        start_time_printBatch = start_time

        # fit each item's factors sequentially (not in parallel)
        for currentItem in range(n_items):

            # get the target column
            y = URM_train[:, currentItem].toarray()

            # set the j-th column of X to zero
            start_pos = URM_train.indptr[currentItem]
            end_pos = URM_train.indptr[currentItem + 1]

            current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
            URM_train.data[start_pos: end_pos] = 0.0

            # fit one ElasticNet model per column
            self.model.fit(URM_train, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values
            nonzero_model_coef_index = self.model.sparse_coef_.indices
            nonzero_model_coef_value = self.model.sparse_coef_.data

            # Check if there are more data points than topK, if so, extract the set of K best values
            if len(nonzero_model_coef_value) > self.topK:
                # Partition the data because this operation does not require to fully sort the data
                relevant_items_partition = np.argpartition(-np.abs(nonzero_model_coef_value), self.topK-1, axis=0)[0:self.topK]
                nonzero_model_coef_index = nonzero_model_coef_index[relevant_items_partition]
                nonzero_model_coef_value = nonzero_model_coef_value[relevant_items_partition]

            similarity_builder.add_data_lists(row_list_to_add=nonzero_model_coef_index,
                                              col_list_to_add=np.ones(len(nonzero_model_coef_index), dtype = np.int32) * currentItem,
                                              data_list_to_add=nonzero_model_coef_value)


            # finally, replace the original values of the j-th column
            URM_train.data[start_pos:end_pos] = current_item_data_backup

            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)


            if time.time() - start_time_printBatch > 300 or currentItem == n_items-1:
                self._print("Processed {} ({:4.1f}%) in {:.2f} {}. Items per second: {:.2f}".format(
                    currentItem+1,
                    100.0* float(currentItem+1)/n_items,
                    new_time_value,
                    new_time_unit,
                    float(currentItem)/elapsed_time))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        self.W_sparse = similarity_builder.get_SparseMatrix()



from multiprocessing import Pool, cpu_count, shared_memory
from functools import partial


def create_shared_memory(a):
    shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
    b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
    b[:] = a[:]
    return shm

@ignore_warnings(category=ConvergenceWarning)
def _partial_fit(URM_train, items, topK, alpha, l1_ratio, positive_only):
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        positive=positive_only,
        fit_intercept=False,
        copy_X=False,
        precompute=True,
        selection='random',
        max_iter=100,
        tol=1e-4
    )
    values, rows, cols = [], [], []

    for currentItem in items:
        y = URM_train[:, currentItem].toarray()

        # Backup dei dati
        start_pos = URM_train.indptr[currentItem]
        end_pos = URM_train.indptr[currentItem + 1]
        backup = URM_train.data[start_pos:end_pos].copy()
        URM_train.data[start_pos:end_pos] = 0.0

        model.fit(URM_train, y)

        nonzero_model_coef_index = model.sparse_coef_.indices
        nonzero_model_coef_value = model.sparse_coef_.data

        if len(nonzero_model_coef_value) > topK:
            relevant_items_partition = np.argpartition(
                -np.abs(nonzero_model_coef_value), topK - 1)[:topK]
            nonzero_model_coef_index = nonzero_model_coef_index[relevant_items_partition]
            nonzero_model_coef_value = nonzero_model_coef_value[relevant_items_partition]

        values.extend(nonzero_model_coef_value)
        rows.extend(nonzero_model_coef_index)
        cols.extend([currentItem] * len(nonzero_model_coef_index))

        # Ripristina i dati
        URM_train.data[start_pos:end_pos] = backup

    return values, rows, cols


class MultiThreadSLIM_SLIMElasticNetRecommender(SLIMElasticNetRecommender):
    def fit(self, alpha=1.0, l1_ratio=0.1, positive_only=True, topK=100, workers=4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        workers=cpu_count()-1
        self.topK = topK

        self.URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)

        item_chunks = np.array_split(np.arange(self.n_items), workers)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Aggiunta della barra di progresso
            with tqdm(total=self.n_items, desc="Progress", unit="items") as pbar:
                futures = [
                    executor.submit(_partial_fit, self.URM_train.copy(), chunk, topK, alpha, l1_ratio, positive_only)
                    for chunk in item_chunks
                ]

                values, rows, cols = [], [], []
                for future, chunk in zip(futures, item_chunks):
                    v, r, c = future.result()
                    values.extend(v)
                    rows.extend(r)
                    cols.extend(c)
                    # Aggiorna la barra di progresso con il numero di item nel chunk
                    pbar.update(len(chunk))

        self.W_sparse = sps.csr_matrix((values, (rows, cols)), shape=(self.n_items, self.n_items), dtype=np.float32)
        self.URM_train = self.URM_train.tocsr()