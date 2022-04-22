#!/usr/bin/env python
# coding: utf-8

# In[2]:


#downloading the file form nltk book download 
import nltk
nltk.download()


# In[3]:


#loading the texts from the from the book
from nltk.book import*


# In[4]:


#loading the text one
text1


# In[5]:


#loading another text 
text2


# In[6]:


#searching text with the help of the function
text1.concordance("monstrous")


# In[26]:


#doing some experiments with the text data to find out the words in the other texts
#concordance is used to find the special text from the text
text3.concordance("lived")


# In[27]:


text2.concordance("affection")


# In[30]:


text2.concordance("lived")


# In[8]:


#append means to add the single item to the already created list.
#finding out the similar text form the dataset text1
text1.similar("monstrous")


# In[9]:


#finding out the similar text form the dataset form text2
text2.similar("monstrous")


# In[28]:


#observe that autin used this word more differently than the melville: for her the monstrous has positive connotations
#lets check it out for some other words,
text2.similar("affection")
#as we can see the words similar to the sad are more present the positive words in the sentence


# In[22]:


text2.similar("happy")


# In[13]:


#the term common contexts allows us to examine just the contexts that are shared by the two or more words
text2.common_contexts(["monstrous","very"])


# In[31]:


#finding out the common context of the different words
text2.common_contexts(["affection","lived"])


# In[ ]:


#trying to plot the disperssion plot and seeing the result

