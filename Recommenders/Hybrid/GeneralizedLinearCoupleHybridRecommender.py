# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from Recommenders.BaseRecommender import BaseRecommender

class GeneralizedNormalizedLinearCoupleHybridRecommender(BaseRecommender):
    """
    This recommender merges N recommendes by weighting their ratings
    """

    RECOMMENDER_NAME = "GeneralizedNormalizedLinearCoupleHybridRecommender"

    def __init__(self, URM_train, recommenders: list, verbose=True):
        self.RECOMMENDER_NAME = ''
        assert len(recommenders) == 2
        for recommender in recommenders:
            self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + 'HybridNormalizedRecommender'

        super(GeneralizedNormalizedLinearCoupleHybridRecommender, self).__init__(URM_train, verbose=verbose)

        self.recommenders = recommenders

    def fit(self, alpha=None):
        self.alpha = alpha

    def save_model(self, folder_path, file_name=None):
        pass

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        pre_res = self.recommenders[0]._compute_item_score(user_id_array,items_to_compute)
        mini = pre_res.min()
        maxi = pre_res.max()
        pre_res_scaled = pre_res - mini / (maxi - mini)

        result = self.alpha*pre_res_scaled
        for index in range(1,len(self.alphas)):
            pre_res = self.recommenders[index]._compute_item_score(user_id_array,items_to_compute)
            mini = pre_res.min()
            maxi = pre_res.max()
            pre_res_scaled = pre_res - mini / (maxi - mini)
            result = result + (1-self.alpha) * pre_res_scaled
        return result

class GeneralizedLinearCoupleHybridRecommender(BaseRecommender):
    """
    This recommender merges N recommendes by weighting their ratings
    """

    RECOMMENDER_NAME = "GeneralizedLinearCoupleHybridRecommender"

    def __init__(self, URM_train, recommenders: list, verbose=True):
        self.RECOMMENDER_NAME = ''
        assert len(recommenders) == 2
        for recommender in recommenders:
            self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + 'HybridRecommender'

        super(GeneralizedLinearCoupleHybridRecommender, self).__init__(URM_train, verbose=verbose)

        self.recommenders = recommenders

    def fit(self, alpha=None):
        self.alpha = alpha

    def save_model(self, folder_path, file_name=None):
        pass

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        result = self.alpha*self.recommenders[0]._compute_item_score(user_id_array,items_to_compute)
        result += (1-self.alpha)*self.recommenders[1]._compute_item_score(user_id_array,items_to_compute)
        return result