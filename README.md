# cf_recommender
Collaborative filtering recommendation system


This is a Python implementation of a collaborative filtering recommender system. It is based on the MovieLens dataset, but readily applicable to other similar settings.

The data can be obtained from http://grouplens.org/datasets/movielens/.

v0.1
Implemented:

  - Base class
  - BaselinePredictor class
  - LatentFactorVariable class


Base class 
  - basic data import
  - transformation of data into suitable formats


Baseline class
  - model of the type r = r_m + b_u + b_i, where r_m is the average score among all items, b_u user specific bias, b_i item specific bias
  - implemented functions for fitting and evaluating the model (including from external data)


LatentFactorVariable1 class
  - model of the type r = r_m + b_u + b_i + q_i.T * p_u, where r_m is the average score among all items, b_u user specific bias, b_i item specific bias, q_i item factors vector, p_u user factors vector








