import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
ewa1 = pd.read_csv("E:/Data Science 18012022/Data Mining K Means/EastWestAirlines.csv")

ewa1.describe()
ewa1.info()

ewa = ewa1.drop(["ID#"], axis=1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(ewa.iloc[:, 0:])
df_norm.describe()

#finding outliers in eastwestairlines#

# Detection of outliers (find limits for salary based on IQR)
IQR = ewa['Balance'].quantile(0.75) - ewa['Balance'].quantile(0.25)
lower_limit = ewa['Balance'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Balance'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa.Balance)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Balance'])

ewa['Balance'] = winsor.fit_transform(ewa[['Balance']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa.Balance)

# Detection of outliers (find limits for salary based on IQR)
IQR = ewa['Bonus_miles'].quantile(0.75) - ewa['Bonus_miles'].quantile(0.25)
lower_limit = ewa['Bonus_miles'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Bonus_miles'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa.Qual_miles)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Bonus_miles'])

ewa['Bonus_miles'] = winsor.fit_transform(ewa[['Bonus_miles']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa.Bonus_miles)


# Detection of outliers (find limits for salary based on IQR)
IQR = ewa['Bonus_trans'].quantile(0.75) - ewa['Bonus_trans'].quantile(0.25)
lower_limit = ewa['Bonus_trans'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Bonus_trans'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa.Qual_miles)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Bonus_trans'])

ewa['Bonus_trans'] = winsor.fit_transform(ewa[['Bonus_trans']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa.Bonus_trans)

# Detection of outliers (find limits for salary based on IQR)
IQR = ewa['Flight_miles_12mo'].quantile(0.75) - ewa['Flight_miles_12mo'].quantile(0.25)
lower_limit = ewa['Flight_miles_12mo'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Flight_miles_12mo'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa.Qual_miles)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Flight_miles_12mo'])

ewa['Flight_miles_12mo'] = winsor.fit_transform(ewa[['Flight_miles_12mo']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa.Flight_miles_12mo)


# Detection of outliers (find limits for salary based on IQR)
IQR = ewa['Flight_trans_12'].quantile(0.75) - ewa['Flight_trans_12'].quantile(0.25)
lower_limit = ewa['Flight_trans_12'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Flight_trans_12'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa.Qual_miles)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Flight_trans_12'])

ewa['Flight_trans_12'] = winsor.fit_transform(ewa[['Flight_trans_12']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa.Flight_trans_12)


# outlier analysis for the dataset 
IQR = ewa['Days_since_enroll'].quantile(0.75) - ewa['Days_since_enroll'].quantile(0.25)
lower_limit = ewa['Days_since_enroll'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Days_since_enroll'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa.Qual_miles)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Days_since_enroll'])

ewa['Days_since_enroll'] = winsor.fit_transform(ewa[['Days_since_enroll']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa.Days_since_enroll)

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

ewa1 = df_norm.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]
ewa.head()

ewa.iloc[:, 0:].groupby(df_norm.clust).mean()

ewa1.to_csv("Eastwestairlines.csv", encoding = "utf-8")

import os
os.getcwd()

