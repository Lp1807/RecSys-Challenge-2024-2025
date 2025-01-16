# Recommender System 2024 Challenge - Polimi
<p align="center">
  <img width="100%" src="https://i.imgur.com/tm9mSuM.png" alt="header" />
</p>
<p align="center">
    <img src="https://i.imgur.com/mPb3Qbd.gif" width="180" alt="Politecnico di Milano"/>
</p>
This repo contains the code and the data used in the [Polimi's Recsys Challenge 2024](https://www.kaggle.com/competitions/recommender-system-2024-challenge-polimi)
<br>
The goal of the competition was to create a recommender system for books, providing 10 recommendations for each user.

## Results

* MAP@10 - private: &nbsp;0.10393
* MAP@10 - public: &nbsp;0.10249
* Ranked 5th

## Goal
The application domain is book recommendation. The datasets we provide contains interactions of users with books. The main goal of the competition is to discover which new books a user will interact with.

# Dataset
The datasets includes around 1.9 M interactions, 35k users, 38k items (books) as well as 94k item features.
The training-test split is done via random holdout, 80% training, 20% test.
The goal is to recommend a list of 10 potentially relevant new items for each user. MAP@10 is used for evaluation.

## Evaluation
The evaluation metric for this competition was MAP@10.

# Recommender
<p align="center">
  <img width="100%" src="images/recommender.png" alt="header" />
</p>
Our best recommender is a hybrid composed of
* SLIM ElasticNet
* RP3Beta
* IALS

Some interesting notebooks can be found in [Notebooks](/MyNotebooks)

## Hyperparameter Tuning
Hyperparameter tuning played a pivotal role in enhancing the performance of our recommender system. To achieve this, we leveraged Optuna for its efficient optimization capabilities. Initially, 80% of the user-interaction dataset was used to conduct multiple Optuna runs. Once an approximate range of optimal solutions was identified, we refined the hyperparameter space and restarted the trials, accelerating convergence while maintaining precision. Additionally, whenever feasible within the constraints of time and computational resources, k-fold cross-validation was employed to further validate the robustness and reliability of the tuned hyperparameters.

<!-- ## Explored models
A full range of models was explored and some in the local  -->

<!-- ## Exploring XGBoost

In our pursuit of maximizing performance, we explored the possibility of integrating XGBoost into our recommender system. This endeavor involved retraining all our models and optimizing them for Recall@25 instead of MAP@10, aiming to leverage the strengths of XGBoost for improved recommendation accuracy. However, despite our efforts, we consistently obtained inferior results compared to our baseline hybrid model. This outcome suggests that we may have made some mistakes in the implementation or parameter tuning of the XGBoosted model. Nevertheless, this experience has provided valuable insights, and we remain commited to revisiting and refining the XGBoosted model in the future. An updated and correct version of the XGBoosted model may emerge in the coming months, reflecting our ongoing commitment to enhancing the performance and robustness of our recommender system. -->

<!-- ## Team
* [Jacopo Piazzalunga](https://github.com/Jacopopiazza)
* [Davide Salonico](https://github.com/DavideSalonico) -->

## Credits
This repository is based on [Official Polimi Recommender Systems repository](https://github.com/recsyspolimi/RecSys_Course_AT_PoliMi)