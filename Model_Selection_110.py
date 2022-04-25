#!/usr/bin/env python
# coding: utf-8

# ## Summary: Compare model results and final model selection
# 
# Using the Titanic dataset from [this](https://www.kaggle.com/c/titanic/overview) Kaggle competition.
# 
# In this section, we will do the following:
# 1. Evaluate all of our saved models on the validation set
# 2. Select the best model based on performance on the validation set
# 3. Evaluate that model on the holdout test set

# ### Read in Data

# In[1]:


import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time

val_features = pd.read_csv('../../../val_features.csv')
val_labels = pd.read_csv('../../../val_labels.csv', header=None)

te_features = pd.read_csv('../../../test_features.csv')
te_labels = pd.read_csv('../../../test_labels.csv', header=None)


# ### Read in Models

# In[2]:


models = {}

for mdl in ['LR', 'SVM', 'MLP', 'RF', 'GB']:
    models[mdl] = joblib.load('../../../{}_model.pkl'.format(mdl))


# In[3]:


models


# ### Evaluate models on the validation set
# 
# ![Evaluation Metrics](../../img/eval_metrics.png)

# In[4]:


#we are defining a function to find the model accurary with respect to different model and test then on different metrics
def evaluate_model(name, model, features, labels):
    start = time()
    pred = model.predict(features)
    end = time()
    accuracy = round(accuracy_score(labels, pred), 3)
    precision = round(precision_score(labels, pred), 3)
    recall = round(recall_score(labels, pred), 3)
    print('{} -- Accuracy: {} / Precision: {} / Recall: {} / Latency: {}ms'.format(name,
                                                                                   accuracy,
                                                                                   precision,
                                                                                   recall,
                                                                                   round((end - start)*1000, 1)))


# In[10]:


#we are using the for loop to find the best model on our training set
for name, mdl in models.items():
    evaluate_model(name, mdl, val_features, val_labels)
    
#we can see it time is not the estimator then our RF model works best have here
#we need to have alot of converstation in order to which one to select based on our need and time 


# ### Evaluate best model on test set

# In[11]:


#we have selected the random forest as our best model and now we are running it on out test set to find the metrics of performance
evaluate_model('Random Forest', models['RF'], te_features, te_labels)


# In[ ]:




