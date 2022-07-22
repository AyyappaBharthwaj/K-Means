import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 
ewa1 = pd.read_csv("E:/Data Science 18012022/Hierarcial Clustering/AutoInsurance.csv")
ewa1.isnull
ewa1.describe()
ewa1.info()

#### Drop Columns
ewa = ewa1.drop(["Customer","State","Effective To Date"], axis=1)

ewa.dtypes

#@create dummies
df_new=pd.get_dummies(ewa)
df_new1=pd.get_dummies(ewa , drop_first=True)

#Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x) 

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df_new1.iloc[:, :])
df_norm.describe()

# Generating random uniform numbers 
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
df_xy = pd.DataFrame(columns=["X","Y"])
df_xy.X = X
df_xy.Y = Y

df_xy.plot(x="X", y="Y", kind = "scatter")

model1 = KMeans(n_clusters = 3).fit(df_xy)

df_xy.plot(x = "X", y = "Y", c = model1.labels_, kind="scatter", s = 10, cmap = plt.cm.coolwarm)

df_norm.describe()

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

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df_norm['clust'] = mb # creating a  new column and assigning it to new column 

ewa.head()
df_norm.head()

ewa1 = df_norm.iloc[:,[47,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46]]
ewa.head()

ewa.iloc[:, 0:].groupby(df_norm.clust).mean()

ewa1.to_csv("Auto insurance.csv", encoding = "utf-8")

import os
os.getcwd()

