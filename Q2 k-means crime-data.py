# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 11:45:21 2022

@author: HARSHITH REDDY
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:11:48 2022

@author: HARSHITH REDDY
"""

import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
ewa1 = pd.read_csv("E:/360 for classes/360 digit Assignments/dataset assignment clus/crime_data.csv")

ewa1.describe()
ewa1.info()

ewa = ewa1.drop(["State"], axis=1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(ewa.iloc[:, 0:])
df_norm.describe()

#finding outliers in eastwestairlines#

# Detection of outliers (find limits for salary based on IQR)
IQR = ewa['Murder'].quantile(0.75) - ewa['Murder'].quantile(0.25)
lower_limit = ewa['Murder'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Murder'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa.Murder)



# Detection of outliers (find limits for salary based on IQR)
IQR = ewa['Assault'].quantile(0.75) - ewa['Assault'].quantile(0.25)
lower_limit = ewa['Assault'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Assault'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa.Assault)




# Detection of outliers (find limits for salary based on IQR)
IQR = ewa['UrbanPop'].quantile(0.75) - ewa['UrbanPop'].quantile(0.25)
lower_limit = ewa['UrbanPop'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['UrbanPop'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa.UrbanPop)



# Detection of outliers (find limits for salary based on IQR)
IQR = ewa['Rape'].quantile(0.75) - ewa['Rape'].quantile(0.25)
lower_limit = ewa['Rape'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Rape'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa.Rape)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Rape'])

ewa['Rape'] = winsor.fit_transform(ewa[['Rape']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa.Rape)


import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# Generating random uniform numbers 
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
df_xy = pd.DataFrame(columns=["X","Y"])
df_xy.X = X
df_xy.Y = Y

df_xy.plot(x="X", y="Y", kind = "scatter")

model1 = KMeans(n_clusters = 3).fit(df_xy)

df_xy.plot(x = "X", y = "Y", c = model1.labels_, kind="scatter", s = 10, cmap = plt.cm.coolwarm)

# Kmeans on University Data set 


ewa.describe()


# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(ewa.iloc[:, 0:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df_norm['clust'] = mb # creating a  new column and assigning it to new column 

ewa1.head()
df_norm.head()

ewa1 = df_norm.iloc[:,[4,0,1,2,3]]
df_norm.head()

df_norm.iloc[:, 0:].groupby(df_norm.clust).mean()

ewa1.to_csv("crime_data.csv", encoding = "utf-8")

import os
os.getcwd()
