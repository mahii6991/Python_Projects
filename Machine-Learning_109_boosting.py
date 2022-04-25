#!/usr/bin/env python
# coding: utf-8

# ## Boosting: Hyperparameters
# 
# Import [`GradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) and [`GradientBoostingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) from `sklearn` and explore the hyperparameters.

# ### Import Boosting Algorithm for Classification & Regression

# In[13]:


#boosting is an ensemble method that aggregate a number of weak model to create a one strong model
#It iteratively built the weak model and learn from it to built the better model.
#Boosting effectively learn from its mistakes with each iteration.so in random forest each of the tree is 
#is built differently but same is not the case with the 
#boosting is of different type what we were going to use is gradient boosting

#when to use it
#can be used for classification and contineous
#useful or nearly any type of problem
#interested in significance of predictors

#when to not use it
#transparency is important
#limited time and computing power
# data is really noisy
#tendency to overfit

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

print(GradientBoostingClassifier())
print(GradientBoostingRegressor())


# In[14]:


import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

tr_features = pd.read_csv('train_features.csv')
tr_labels = pd.read_csv('train_labels.csv', header=None)


# In[15]:


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))


# In[17]:


gb = GradientBoostingClassifier()
parameters = {
    'n_estimators': [5,50,250,500],
    'max_depth': [1,3,5,7,9],
    'learning_rate': [0.01,0.1,1,10,100]
}

cv= GridSearchCV(gb, parameters, cv=5)
cv.fit(tr_features,tr_labels.values.ravel())

print_results(cv)


# In[18]:


joblib.dump(cv.best_estimator_,'GB_model.pkl')


# In[ ]:


gb = GradientBoostingClassifier()
parameters = {
    'n_estimators': [],
    'max_depth': [],
    'learning_rate': []
}

