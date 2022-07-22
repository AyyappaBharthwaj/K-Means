import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
ewa1 = pd.read_csv("E:/Data Science 18012022/Data Mining K Means/Insurance Dataset.csv")

ewa1.describe()
ewa1.info()
ewa = ewa1
# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(ewa.iloc[:, 0:])
df_norm.describe()


#finding outliers in eastwestairlines#

# Detection of outliers (find limits for Premiums Paid based on IQR)
IQR = ewa['Premiums Paid'].quantile(0.75) - ewa['Premiums Paid'].quantile(0.25)
lower_limit = ewa['Premiums Paid'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Premiums Paid'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa['Premiums Paid'])

# Trimming Technique
# let's flag the outliers in the data set
outliers_ewa = np.where(ewa['Premiums Paid'] > upper_limit, True, np.where(ewa['Premiums Paid'] < lower_limit, True, False))
ewa = ewa.loc[~(outliers_ewa), ]
ewa.shape

# let's explore outliers in the trimmed dataset
sns.boxplot(ewa['Premiums Paid'])


# Detection of outliers (find limits for Age based on IQR)
IQR = ewa['Age'].quantile(0.75) - ewa['Age'].quantile(0.25)
lower_limit = ewa['Age'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Age'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa['Age'])

#### No Outliers##############

# Detection of outliers (find limits for Days to Renew based on IQR)
IQR = ewa['Days to Renew'].quantile(0.75) - ewa['Days to Renew'].quantile(0.25)
lower_limit = ewa['Days to Renew'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Days to Renew'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa['Days to Renew'])

#### No Outliers##############

# Detection of outliers (find limits for Claims made based on IQR)
IQR = ewa['Claims made'].quantile(0.75) - ewa['Claims made'].quantile(0.25)
lower_limit = ewa['Claims made'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Claims made'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa['Claims made'])

###############  Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Claims made'])

ewa['Claims made'] = winsor.fit_transform(ewa[['Claims made']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa['Claims made'])
######################################################

# Detection of outliers (find limits for Income based on IQR)
IQR = ewa['Income'].quantile(0.75) - ewa['Income'].quantile(0.25)
lower_limit = ewa['Income'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Income'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa['Income'])

######## No Outliers #################################

###### K Means Clustering##############################
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

ewa.head()
df_norm.head()

ewa1 = df_norm.iloc[:,[5,0,1,2,3,4]]
ewa.head()

ewa.iloc[:, 0:].groupby(df_norm.clust).mean()

ewa1.to_csv("Insurancedataset.csv", encoding = "utf-8")

import os
os.getcwd()














