#!/usr/bin/env python
# coding: utf-8

# 模型融合

# 1. https://www.kaggle.com/filemide/distance-criskiev-hyparam-cont-1-662
# 2. https://www.kaggle.com/criskiev/distance-is-all-you-need-lb-1-481
# 3. https://www.kaggle.com/marcelotamashiro/lgb-public-kernels-plus-more-features
# 4. https://www.kaggle.com/scaomath/no-memory-reduction-workflow-for-each-type-lb-1-28
# 5. https://www.kaggle.com/fnands/1-mpnn/output?scriptVersionId=18233432
# 6. https://www.kaggle.com/harshit92/fork-from-kernel-1-481
# 7. https://www.kaggle.com/xwxw2929/keras-neural-net-and-distance-features
# 8. https://www.kaggle.com/marcogorelli/criskiev-s-distances-more-estimators-groupkfold?scriptVersionId=18843561
# 9. https://www.kaggle.com/toshik/schnet-starter-kit

# In[1]:


import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[2]:


sub_path = "../input/models"
all_files = os.listdir(sub_path)
all_files


# In[3]:


all_files.remove('submission-1.701.csv')
all_files.remove('submission-1.643.csv')
all_files.remove('submission-1.481.csv')
all_files.remove('submission-1.302.csv')
all_files.remove('submission-1.619.csv')
all_files.remove('submission-1.662.csv')
all_files.remove('submission-1.696.csv')
all_files.remove('submission-1.780.csv')
all_files.remove('submission-1.708.csv')
all_files.remove('submission-1.714.csv')


# In[4]:


all_files


# In[5]:


import warnings
warnings.filterwarnings("ignore")
outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "mol" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)
concat_sub.head()
ncol = concat_sub.shape[1]


# In[6]:


#检查相关性
concat_sub.iloc[:,1:].corr()


# In[7]:


corr = concat_sub.iloc[:,1:].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


f, ax = plt.subplots(figsize=(11, 9))

#绘制热力图
sns.heatmap(corr, mask=mask, cmap='prism', vmin=0.96, center=0, square=True, linewidths=1, annot=True, fmt='.4f')


# In[8]:


concat_sub['m_max'] = concat_sub.iloc[:, 1:].max(axis=1)
concat_sub['m_min'] = concat_sub.iloc[:, 1:].min(axis=1)
concat_sub['m_median'] = concat_sub.iloc[:, 1:].median(axis=1)


# In[9]:


concat_sub.describe()


# In[10]:


concat_sub.head(10)


# In[11]:


cutoff_lo = 0.8
cutoff_hi = 0.2


# In[12]:


concat_sub['scalar_coupling_constant'] = concat_sub['m_median']
concat_sub[['id', 'scalar_coupling_constant']].to_csv('stack_median.csv', 
                                        index=False, float_format='%.6f')

