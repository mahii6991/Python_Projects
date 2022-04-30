#!/usr/bin/env python
# coding: utf-8

# ## Explore The Data: What Data Are We Using?
# 
# Using the Titanic dataset from [this](https://www.kaggle.com/c/titanic/overview) Kaggle competition.
# 
# This dataset contains information about 891 people who were on board the ship when departed on April 15th, 1912. As noted in the description on Kaggle's website, some people aboard the ship were more likely to survive the wreck than others. There were not enough lifeboats for everybody so women, children, and the upper-class were prioritized. Using the information about these 891 passengers, the challenge is to build a model to predict which people would survive based on the following fields:
# 
# - **Name** (str) - Name of the passenger
# - **Pclass** (int) - Ticket class (1st, 2nd, or 3rd)
# - **Sex** (str) - Gender of the passenger
# - **Age** (float) - Age in years
# - **SibSp** (int) - Number of siblings and spouses aboard
# - **Parch** (int) - Number of parents and children aboard
# - **Ticket** (str) - Ticket number
# - **Fare** (float) - Passenger fare
# - **Cabin** (str) - Cabin number
# - **Embarked** (str) - Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

# ### Read In Data

# In[1]:


# Read in the data from the data folder
import pandas as pd

titanic = pd.read_csv('titanic.csv')
titanic.head()


# In[4]:


# Check the number of rows and columns in the data
print("the length of the rows is:",len(titanic))
print("the number of column is:",titanic.shape[1])


# In[ ]:


# Check the type of data stored in each column


# In[5]:


titanic.shape[0]
titanic.shape[1]


# In[9]:


# this is the way to find the shape of the dataframe
titanic.shape


# In[10]:


#finding the datatype of the column
titanic.dtypes


# In[11]:


titanic['Survived'].value_counts()
#so we can see that out of the 891 people that were present on the ship only 342 survived.


# In[ ]:


#as we have seen in our predicting class the the number of person servived is less than
#the number of person those were alive.so in this case we can say that our feature class is imbalanced in this case.
#what should we need to do if we face this kind of situation, we can downscale it by using the functions


# In[12]:


#next thing we will do is to drop the categorical featrure form it
# Drop all categorical features
cat_feat = ['PassengerId', 'Name', 'Ticket', 'Sex', 'Cabin', 'Embarked']


# In[13]:


titanic.head()


# In[15]:


#we are going to drop the features that we are selected in the above column
titanic.drop(cat_feat,axis=1)#we only left with the contineouse variable


# # Explore continuous features

# In[16]:


#looking at the general distribution of the featrues
titanic.describe()

#as we can see the age class contain some less variable in them, it means that there are some missing values in it.
#and also we can see that the features class is pretty unbalanced, and in the classification problem the balance matters
#for example our dataset contains the 99% of the variable that are not_survived then in 99% of the time the model will perdict that class


# In[17]:


#now we will going to look at the correlation matrix in the dataset
#finding the correlation matrix
titanic.corr()


# In[19]:


#finding the heatmap of the correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(titanic.corr(), annot = True)
#we need to understand that we need to look at both the correlation vales be it the 
#positive one or the negative one. because the negative correlation is also the correlation
#we found the there is the correlation between the survived and fare variable


# In[20]:


# Look at fare by different passenger class levels
titanic.groupby('Pclass')['Fare'].describe()


# In[31]:


#we will going to run the t-test to find the difference in the mean of the two variables
#so basically we have written a function to calculate the t-test,between different groups
def describe_cont_feature(feature):
    print('\n*** Results for {} ***'.format(feature))
    print(titanic.groupby('Survived')[feature].describe())
    #print(ttest(feature))
    
def ttest(feature):
    survived = titanic[titanic['Survived']==1][feature]
    not_survived = titanic[titanic['Survived']==0][feature]
    tstat, pval = stats.ttest_ind(survived, not_survived, equal_var=False)
    print('t-statistic: {:.1f}, p-value: {:.3}'.format(tstat, pval))
    print(ttest(feature))


# In[32]:


# Look at the distribution of each feature at each level of the target variable
# Look at the distribution of each feature at each level of the target variable
for feature in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:
    describe_cont_feature(feature)


# In[33]:


# Look at the average value of each feature based on whether Age is missing
titanic.groupby(titanic['Age'].isnull()).mean()

#need to understand what does he mean by this result


# In[38]:


# Read in our data
#now we will start to laod some libraries which we did'nt load earlier
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


#we will start by plotting the contineous variable
titanic = pd.read_csv('titanic.csv',
                      usecols=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'])
titanic.head()


# In[39]:


#need to plot the countineous features
# Plot overlaid histograms for continuous features
for i in ['Age', 'Fare']:
    died = list(titanic[titanic['Survived'] == 0][i].dropna())
    survived = list(titanic[titanic['Survived'] == 1][i].dropna())
    xmin = min(min(died), min(survived))
    xmax = max(max(died), max(survived))
    width = (xmax - xmin) / 40
    sns.distplot(died, color='r', kde=False, bins=np.arange(xmin, xmax, width))
    sns.distplot(survived, color='g', kde=False, bins=np.arange(xmin, xmax, width))
    plt.legend(['Did not survive', 'Survived'])
    plt.title('Overlaid histogram for {}'.format(i))
    plt.show()


# In[40]:


# Generate categorical plots for ordinal features
for col in ['Pclass', 'SibSp', 'Parch']:
   sns.catplot(x=col, y='Survived', data=titanic, kind='point', aspect=2, )
   plt.ylim(0, 1)


# In[ ]:




