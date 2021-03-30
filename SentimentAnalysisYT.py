#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


comments=pd.read_csv('C:/Users/wpric/Documents/Data Science/1-Youtube Text Data Analysis/UScomments.csv',error_bad_lines=False)


# In[3]:


comments.head()


# In[ ]:





# In[4]:


get_ipython().system('pip install textblob')


# In[5]:


from textblob import TextBlob


# In[6]:


TextBlob('He happy cause he in a movie..').sentiment.polarity


# In[7]:


comments.isna().sum()


# In[8]:


comments.dropna(inplace=True)


# In[9]:


polarity=[]

for i in comments['comment_text']:
    polarity.append(TextBlob(i).sentiment.polarity)


# In[10]:


comments['polarity']=polarity


# In[11]:


comments.head(20)


# Positive Sentiment Analysis Using WordCloud

# In[12]:


comments_positive=comments[comments['polarity']==1]


# In[13]:


comments_positive.shape


# In[14]:


comments_positive.head()


# In[15]:


get_ipython().system('pip install wordcloud')


# In[16]:


from wordcloud import WordCloud,STOPWORDS


# In[17]:


stopwords=set(STOPWORDS)


# In[18]:


total_comments=''.join(comments_positive['comment_text'])


# In[19]:


wordcloud=WordCloud(width=1000, height=500, stopwords=stopwords).generate(total_comments)


# In[20]:


plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# Negative Sentiment Analysis WordCloud

# In[21]:


comments_negative=comments[comments['polarity']==-1]


# In[22]:


comments_negative.shape


# In[23]:


comments_negative.head()


# In[24]:


total_comments=''.join(comments_negative['comment_text'])


# In[25]:


wordcloud=WordCloud(width=1000, height=500, stopwords=stopwords).generate(total_comments)


# In[26]:


plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# Trending Tags Analysis

# In[27]:


videos=pd.read_csv('C:/Users/wpric/Documents/Data Science/1-Youtube Text Data Analysis/USvideos.csv',error_bad_lines=False)


# In[28]:


videos.head()


# In[29]:


tags_complete=''.join(videos['tags'])


# In[30]:


tags_complete


# In[31]:


import re


# In[32]:


tags=re.sub('[^a-zA-Z]','',tags_complete)


# In[33]:


tags


# In[34]:


tags=re.sub(' +','',tags)


# In[ ]:


wordcloud=WordCloud(width=1000,height=500,stopwords=set(STOPWORDS)).generate(tags)


# In[ ]:


plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# In[ ]:





# Regression Plot for views & likes

# In[ ]:


sns.regplot(data=videos,x='views',y='likes')
plt.title('Regression plot for views & likes')


# In[ ]:





# Regression Plot for views & dislikes

# In[ ]:


sns.regplot(data=videos,x='views',y='dislikes')
plt.title('Regression plot for views & dislikes')


# In[ ]:





# In[ ]:


df_corr=videos[['views','likes','dislikes']]


# In[ ]:


df_corr.corr()


# In[ ]:


sns.heatmap(df_corr.corr(),annot=True)

