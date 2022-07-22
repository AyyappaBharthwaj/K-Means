import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
ewa1 = pd.read_csv("E:/Data Science 18012022/Data Mining K Means/Telco_customer_churn.csv")
ewa1.isnull
ewa1.describe()
ewa1.info()

ewa = ewa1.drop(["Customer ID","Count","Quarter"], axis=1)


################ Finding Outliers #########################
# Finding Outliers in Avg_Monthly_Long_Distance_Charges
ewa['Avg Monthly Long Distance Charges']

# let's find outliers in Avg_Monthly_Long_Distance_Charges
sns.boxplot(ewa['Avg Monthly Long Distance Charges'])

# No Outliers detected in Avg_Monthly_Long_Distance_Charges

#################################################################

# Finding Outliers in Avg_Monthly_Long_Distance_Charges################

ewa['Avg Monthly GB Download']

# let's find outliers in Avg_Monthly_Long_Distance_Charges
sns.boxplot(ewa['Avg Monthly GB Download'])

##### Outliers Found in Avg_Monthly_GB_Download#####################

# Detection of outliers (find limits for Avg Monthly GB Download based on IQR)
IQR = ewa['Avg Monthly GB Download'].quantile(0.75) - ewa['Avg Monthly GB Download'].quantile(0.25)
lower_limit = ewa['Avg Monthly GB Download'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Avg Monthly GB Download'].quantile(0.75) + (IQR * 1.5)

# Trimming Technique
# let's flag the outliers in the data set
outliers_ewa = np.where(ewa['Avg Monthly GB Download'] > upper_limit, True, np.where(ewa['Avg Monthly GB Download'] < lower_limit, True, False))
ewa = ewa.loc[~(outliers_ewa), ]
ewa.shape

# let's explore outliers in the trimmed dataset
sns.boxplot(ewa['Avg Monthly GB Download'])

### Flagged out Outliers through Trimming Process ##################

# Finding Outliers in Monthly_Charge
ewa['Monthly Charge']

# let's find outliers in Monthly_Charge
sns.boxplot(ewa['Monthly Charge'])
########### No Outliers Found in Monthly_Charge##############

# Finding Outliers in Total_Charges
ewa['Total Charges']

# let's find outliers in Total_Charges
sns.boxplot(ewa['Total Charges'])
########### No Outliers Found in Total_Charges##############

# Finding Outliers in Total_Refunds
ewa['Total Refunds']

# let's find outliers in Total_Refunds
sns.boxplot(ewa['Total Refunds'])
###### Outliers found in Total_Refunds ######################

# Detection of outliers (find limits for Total Refunds based on IQR)
IQR = ewa['Total Refunds'].quantile(0.75) - ewa['Total Refunds'].quantile(0.25)
lower_limit = ewa['Total Refunds'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Total Refunds'].quantile(0.75) + (IQR * 1.5)

############### Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Total Refunds'])

ewa['Total Refunds'] = winsor.fit_transform(ewa[['Total Refunds']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa['Total Refunds'])
###############################################################

# Finding Outliers in Total_Extra_Data_Charges
ewa['Total Extra Data Charges']

# let's find outliers in Total_Extra_Data_Charges
sns.boxplot(ewa['Total Extra Data Charges'])
###### Outliers found in Total_Extra_Data_Charges ######################

# Detection of outliers (find limits for Total_Extra_Data_Charges based on IQR)
IQR = ewa['Total Extra Data Charges'].quantile(0.75) - ewa['Total Extra Data Charges'].quantile(0.25)
lower_limit = ewa['Total Extra Data Charges'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Total Extra Data Charges'].quantile(0.75) + (IQR * 1.5)

############### Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Total Extra Data Charges'])

ewa['Total Extra Data Charges'] = winsor.fit_transform(ewa[['Total Extra Data Charges']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa['Total Extra Data Charges'])
#####################################################################
#####################################################################

# Finding Outliers in Total_Long_Distance_Charges
ewa['Total Long Distance Charges']

# let's find outliers in Total_Long_Distance_Charges
sns.boxplot(ewa['Total Long Distance Charges'])
###### Outliers found in Total_Long_Distance_Charges ######################

IQR = ewa['Total Long Distance Charges'].quantile(0.75) - ewa['Total Long Distance Charges'].quantile(0.25)
lower_limit = ewa['Total Long Distance Charges'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Total Long Distance Charges'].quantile(0.75) + (IQR * 1.5)

############### Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Total Long Distance Charges'])

ewa['Total Long Distance Charges'] = winsor.fit_transform(ewa[['Total Long Distance Charges']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa['Total Long Distance Charges'])

###################################################################
###################################################################

# Finding Outliers in Total_Revenue
ewa['Total Revenue'] 

# let's find outliers in Total_Long_Distance_Charges
sns.boxplot(ewa['Total Revenue'])
###### Outliers found in Total_Long_Distance_Charges ######################

IQR = ewa['Total Revenue'].quantile(0.75) - ewa['Total Revenue'].quantile(0.25)
lower_limit = ewa['Total Revenue'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Total Revenue'].quantile(0.75) + (IQR * 1.5)

############### Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Total Revenue'])

ewa['Total Revenue'] = winsor.fit_transform(ewa[['Total Revenue']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa['Total Revenue'])
ewa.shape

############################################################################


ewa.columns # column names
ewa.shape # will give u shape of the dataframe


# Create dummy variables
ewa_new = pd.get_dummies(ewa)
ewa_new_1 = pd.get_dummies(ewa, drop_first = True)
# we have created dummies for all categorical columns

##### One Hot Encoding works
ewa.columns
ewa.shape
df = ewa.iloc[:,0:]
df.shape

from sklearn.preprocessing import OneHotEncoder
# Creating instance of One Hot Encoder
enc = OneHotEncoder() # initializing method

enc_df = pd.DataFrame(enc.fit_transform(df.iloc[:, 0:]).toarray())
enc_df.shape

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

enc_df.describe()

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(enc_df)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(enc_df)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
enc_df['clust'] = mb # creating a  new column and assigning it to new column 

ewa.head()
enc_df.head()

ewa1 = enc_df.iloc[:,[27,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]]
ewa.head()

ewa.iloc[:, 0:].groupby(enc_df.clust).mean()

ewa1.to_csv("Telco Customer Churn.csv", encoding = "utf-8")

import os
os.getcwd()
