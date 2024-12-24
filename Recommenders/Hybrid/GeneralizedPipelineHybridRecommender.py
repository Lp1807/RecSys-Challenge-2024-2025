# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from Recommenders.BaseRecommender import BaseRecommender
import numpy as np
from scipy.sparse import csr_matrix
import gc
from sklearn.preprocessing import MinMaxScaler


class GeneralizedPipelineHybridRecommender(BaseRecommender):
    """
    This recommender merges N recommendes by weighting their ratings
    """

    RECOMMENDER_NAME = "GeneralizedPipelineHybridRecommender"

    def __init__(self, URM_train, first_model, params_firstmodel, second_model, params_secondmodel, threshold=0.85, verbose=True):
        self.RECOMMENDER_NAME = ''
        self.first_model = first_model
        self.second_model = second_model

        self.threshold = threshold

        self.params_firstmodel = params_firstmodel
        self.params_secondmodel = params_secondmodel

        self.RECOMMENDER_NAME = first_model.RECOMMENDER_NAME + "_" + second_model.RECOMMENDER_NAME

        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + 'PipelineHybridRecommender'

        super(GeneralizedPipelineHybridRecommender, self).__init__(URM_train, verbose=verbose)


    def fit(self):

        # Calculate sparsity
        total_elements = self.URM_train.shape[0] * self.URM_train.shape[1]
        non_zero_elements = self.URM_train.count_nonzero()
        sparsity = 1.0 - (non_zero_elements / total_elements)

        print(f"URM Train Sparsity of the CSR matrix: {sparsity}, #non null: {self.URM_train.nnz}")

        self.first_rec = self.first_model(self.URM_train)
        self.first_rec.fit(**self.params_firstmodel)
        print("Ended training of the first model")

        #self.URM_pipeline = self.URM_train
        self.URM_pipeline = np.zeros((self.URM_train.shape[0], self.URM_train.shape[1]))

        for u in range(self.URM_train.shape[0]):
            self.URM_pipeline[u, self.first_rec.recommend(u)[:self.threshold]] = 1
        #print(f"URM Pipeline shape: {self.URM_train.shape}")
        #maxi = self.URM_pipeline.max()
        ##mini = self.URM_pipeline.min()
        #print(f"URM Pipeline min: {mini}, max: {maxi}")

        #scaler = MinMaxScaler()
        #self.URM_pipeline = scaler.fit_transform(self.URM_pipeline)

        #mini = self.URM_pipeline.min()
        #maxi = self.URM_pipeline.max()
        #print(f"URM Pipeline after normalization min: {mini}, max: {maxi}")

        self.URM_pipeline += self.URM_train

        #mask = self.URM_pipeline > self.threshold

        # Setting elements close to 0.85 to 1 and others to 0
        #self.URM_pipeline[mask] = 1
        #self.URM_pipeline[~mask] = 0

        self.URM_pipeline = csr_matrix(self.URM_pipeline)
        self.URM_pipeline.eliminate_zeros()

        gc.collect()

        # Calculate sparsity
        total_elements = self.URM_pipeline.shape[0] * self.URM_pipeline.shape[1]
        non_zero_elements = self.URM_pipeline.count_nonzero()
        sparsity = 1.0 - (non_zero_elements / total_elements)

        print(self.URM_pipeline.nnz)
        print(f"URM Pipeline Sparsity of the CSR matrix: {sparsity}, #non null: {self.URM_pipeline.nnz}")

        self.second_rec = self.second_model(self.URM_pipeline)
        print("Start training of the second model")
        self.second_rec.fit(**self.params_secondmodel)

        print("Ended training of the second model")

    def _compute_item_score(self, user_id_array, items_to_compute = None):
        return self.second_rec._compute_item_score(user_id_array, items_to_compute)

    def _remove_seen_on_scores(self, user_id, scores):

        assert self.URM_train.getformat() == "csr", "Recommender_Base_Class: URM_train is not CSR, this will cause errors in filtering seen items"

        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

        scores[seen] = -np.inf
        return scores

        