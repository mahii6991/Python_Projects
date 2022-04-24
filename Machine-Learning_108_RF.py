#!/usr/bin/env python
# coding: utf-8

# ## Random Forest: Hyperparameters
# 
# Import [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and [`RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) from `sklearn` and explore the hyperparameters.

# ### Import Random Forest Algorithm for Classification & Regression

# In[2]:


#Random forest
#it is collection of independent decision trees to get more accurate and stable prediction.
#Ensemble methods combine several machine learning models in order to decrease both bias and variance

#When to use it (just a general idea to get)
# on both contineous and categorical data
# instrested in the significance of the predictor
# need a quick benchmark model to test the model
# if you have messy data , with missing values and outliers it works best

#when to not use it
#if you are solving a very complex , novel problem
# transparency is important
#prediction time is important

from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor

print(RandomForestClassifier())
print(RandomForestRegressor())


# In[4]:


#lets talk about the hyperparameter 
'''
the n_estimators controls how many individual decision trees will be built
in other words it controls the width of the model

the max_depth controls how deep each individual decision tree can go
in other words it control the depth of the model

'''

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

tr_features = pd.read_csv('train_features.csv')
tr_labels = pd.read_csv('train_labels.csv', header=None)


# # Hyperparameter tuning

# In[5]:


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))


# In[8]:


rf = RandomForestClassifier()
parameters = {
    'n_estimators': [5,50,250],
    'max_depth': [2 ,4 ,8, 16, 32, None]
}

cv = GridSearchCV(rf , parameters , cv=5)
#fitting the model 
cv.fit(tr_features, tr_labels.values.ravel())

# print the results 
print_results(cv)


# # write out pickled model

# In[9]:


joblib.dump(cv.best_estimator_,'RF_model.pkl')


# In[ ]:




