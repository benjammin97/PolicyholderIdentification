#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.cluster import DBSCAN
import itertools
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
import seaborn as sns


# In[2]:


auto = pd.read_csv("auto_policies_2020.csv")


# In[3]:


auto


# In[4]:


auto_updated = auto.drop(columns = ["date_of_birth","claim_office","area","pol_number","pol_eff_dt","annual_premium","veh_body","gender"])
#Dropping unimportant columns
#Obviously, policy date, claim number, and claim office have nothing to do with the risk of the driver
#I chose not to use date of birth because there is already an age cat which simplifies things
#I chose not to use area because we have the traffic index which is a standard method for traffic which should normalize area results
#I have decided to drop annual premium as well after realizing the premium is the same for all drivers
#After careful consideration, I have decided to drop gender and vehicle body as well due to the large processing costs to my system


# In[5]:


print(auto_updated.isnull().sum()) # checking for missing values
print(auto_updated.shape)


# In[6]:


auto_updated


# In[7]:


auto_updated["numclaims"] = auto_updated["numclaims"].fillna(0) #making sure zero claims are not dropped with na values
auto_updated = auto_updated.dropna() # Dropping rows with missing values
auto_updated = auto_updated.reset_index(drop = True) #resetting the index


# In[8]:


print(auto_updated)


# In[9]:


#introduce the scaler
scaler = MinMaxScaler()
scaler.fit(auto_updated)
# transform data
auto_updated_scaled = scaler.transform(auto_updated)
# print dataset properties before and after scaling
print("transformed shape: {}".format(auto_updated_scaled.shape))
print("per-feature minimum before scaling:\n {}".format(auto_updated.min(axis=0)))
print("per-feature maximum before scaling:\n {}".format(auto_updated.max(axis=0)))
print("per-feature minimum after scaling:\n {}".format(
    auto_updated_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n {}".format(
    auto_updated_scaled.max(axis=0)))


auto_updated_scaled = pd.DataFrame(auto_updated_scaled)


# In[10]:


SSE = [] #storage array
for k in range(2,7):
    kmeans = KMeans(n_clusters = k) #### This line is where we are changing K in our algorithm
    kmeans.fit(auto_updated_scaled)
    SSE.append(kmeans.inertia_) # Inertia is the SSE i.e. compactness
    
results = pd.DataFrame({"k": range(2,7),
                      "SSE/Inertia": SSE})

print(results)
plt.plot(results["k"], results["SSE/Inertia"], linewidth = 3, color = "black") #line on the graph
plt.xlabel("k value")
plt.ylabel("SSE/Inertia")
plt.title("SSE/Inertia vs. k value")


# In[11]:


# Using the silhouette score metric to determine k
range_n_clusters = [2,3,4,5,6]
silhouette_avg = []
#Taking a smaller sample of the data in order to speed up processing
auto_sample = auto_updated_scaled.sample(frac=.1)
for num_clusters in range_n_clusters:
    # initialise kmeans
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(auto_sample)
    cluster_labels = kmeans.labels_
    # silhouette score
    score = silhouette_score(auto_sample, cluster_labels)
    silhouette_avg.append(score)


# In[12]:


# visualizing silhouette scores
plt.plot(range_n_clusters, silhouette_avg, linewidth = 3, color = "black")
plt.xlabel("k value") # x axis label
plt.ylabel("silhouette score") # y axis label
plt.title("silhouette score vs. k value"); # title


# In[13]:


best_kmeans = KMeans(n_clusters = 3) # Rerun k-means algorithm to get the algorithm with the best k result we chose i.e. k = 6
best_kmeans.fit(auto_updated)


# In[14]:


# Creating a column in the original data with the corresponding clusters
auto_updated['kmeans_clusters'] = best_kmeans.labels_ 
auto_updated.sort_values('kmeans_clusters') # sort clusterings to easily read data frame
auto_updated.groupby("kmeans_clusters").describe() # get summary statistics for each cluster.
#Most drivers end up in the low risk category


# In[15]:


print(auto_updated.describe()) #Describe the auto table


# In[16]:


fig, axes = plt.subplots(2, 2, figsize = (20,20)) # Create grid of subplots using Seaborn

ax = sns.boxplot(data = auto_updated, x = "kmeans_clusters", y = "agecat", ax=axes[0,0]) # Define first plot
ax.set_xlabel('kmeans_clusters', fontsize = 20) # Parameters for first plot, making x label bigger
ax.set_ylabel("Age Category", fontsize = 20) # Parameters for first plot, making y label bigger
ax.tick_params(axis='x', labelsize=16)# Parameters for first plot, making x tick marks bigger
ax.tick_params(axis='y', labelsize=16)# Parameters for first plot, making y tick marks bigger

ax = sns.boxplot(data = auto_updated, x = "kmeans_clusters", y = "credit_score" , ax=axes[0,1])
ax.set_xlabel('kmeans_clusters', fontsize = 20)
ax.set_ylabel("Credit Score", fontsize = 20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

ax = sns.boxplot(data = auto_updated, x = "kmeans_clusters", y = "traffic_index", ax=axes[1,0])
ax.set_xlabel('kmeans_clusters', fontsize = 20)
ax.set_ylabel("Traffic Index", fontsize = 20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

ax = sns.boxplot(data = auto_updated, x = "kmeans_clusters", y = "veh_age" , ax=axes[1,1])
ax.set_xlabel('kmeans_clusters', fontsize = 20)
ax.set_ylabel("Vehicle Age", fontsize = 20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)


# In[17]:


fig, axes = plt.subplots(2, 2, figsize = (20,20)) # Create grid of subplots using Seaborn

ax = sns.boxplot(data = auto_updated, x = "kmeans_clusters", y = "veh_value", ax=axes[0,0]) # Define first plot
ax.set_xlabel('kmeans_clusters', fontsize = 20) # Parameters for first plot, making x label bigger
ax.set_ylabel("Vehicle Value", fontsize = 20) # Parameters for first plot, making y label bigger
ax.tick_params(axis='x', labelsize=16)# Parameters for first plot, making x tick marks bigger
ax.tick_params(axis='y', labelsize=16)# Parameters for first plot, making y tick marks bigger

ax = sns.boxplot(data = auto_updated, x = "kmeans_clusters", y = "numclaims" , ax=axes[0,1])
ax.set_xlabel('kmeans_clusters', fontsize = 20)
ax.set_ylabel("Number of Claims", fontsize = 20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

ax = sns.boxplot(data = auto_updated, x = "kmeans_clusters", y = "claimcst0", ax=axes[1,0])
ax.set_xlabel('kmeans_clusters', fontsize = 20)
ax.set_ylabel("Claim Cost", fontsize = 20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)


# In[18]:


fig, axes = plt.subplots(1, 1, figsize = (7.5,7.5)) # Create grid of subplots using Seaborn
#Creating a smaller figure to fit in the presentation
ax = sns.boxplot(data = auto_updated, x = "kmeans_clusters", y = "credit_score")
ax.set_xlabel('kmeans_clusters', fontsize = 20)
ax.set_ylabel("Credit Score", fontsize = 20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
"""
#ax = sns.boxplot(data = auto_updated, x = "kmeans_clusters", y = "numclaims" , ax=axes[0,1])
#ax.set_xlabel('kmeans_clusters', fontsize = 20)
#ax.set_ylabel("Number of Claims", fontsize = 20)
#ax.tick_params(axis='x', labelsize=16)
#ax.tick_params(axis='y', labelsize=16)

ax = sns.boxplot(data = auto_updated, x = "kmeans_clusters", y = "claimcst0", ax=axes[1,0])
ax.set_xlabel('kmeans_clusters', fontsize = 20)
ax.set_ylabel("Claim Cost", fontsize = 20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

ax = sns.boxplot(data = auto_updated, x = "kmeans_clusters", y = "veh_value", ax=axes[1,1]) # Define first plot
ax.set_xlabel('kmeans_clusters', fontsize = 20) # Parameters for first plot, making x label bigger
ax.set_ylabel("Vehicle Value", fontsize = 20) # Parameters for first plot, making y label bigger
ax.tick_params(axis='x', labelsize=16)# Parameters for first plot, making x tick marks bigger
ax.tick_params(axis='y', labelsize=16)# Parameters for first plot, making y tick marks bigger
"""
#I am putting all of these box plots together to point out the most significant factors for the groups in the presentation


# In[19]:


fig, axes = plt.subplots(1, 1, figsize = (7.5,7.5)) # Create grid of subplots using Seaborn
#Creating a smaller figure to fit in the presentation
ax = sns.boxplot(data = auto_updated, x = "kmeans_clusters", y = "numclaims")
ax.set_xlabel('kmeans_clusters', fontsize = 20)
ax.set_ylabel("Number of Claims", fontsize = 20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)


# In[20]:


fig, axes = plt.subplots(1, 1, figsize = (7.5,7.5)) # Create grid of subplots using Seaborn
#Creating a smaller figure to fit in the presentation
ax = sns.boxplot(data = auto_updated, x = "kmeans_clusters", y = "claimcst0")
ax.set_xlabel('kmeans_clusters', fontsize = 20)
ax.set_ylabel("Claim Cost", fontsize = 20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)


# In[21]:


fig, axes = plt.subplots(1, 1, figsize = (7.5,7.5)) # Create grid of subplots using Seaborn
#Creating a smaller figure to fit in the presentation
ax = sns.boxplot(data = auto_updated, x = "kmeans_clusters", y = "veh_value") # Define first plot
ax.set_xlabel('kmeans_clusters', fontsize = 20) # Parameters for first plot, making x label bigger
ax.set_ylabel("Vehicle Value", fontsize = 20) # Parameters for first plot, making y label bigger
ax.tick_params(axis='x', labelsize=16)# Parameters for first plot, making x tick marks bigger
ax.tick_params(axis='y', labelsize=16)# Parameters for first plot, making y tick marks bigger


# In[22]:


sil_score = [] # Create an empty list to store the silhouette score

eps = np.arange(0.01,1.02,0.5) # This is creating a list of values to test as opposed to me manually typing each value
minpts = np.arange(2,7,1) # This is creating a list of values to test as opposed to me manually typing each value

for ep, minpt in itertools.product(eps, minpts): # Using the itertools package to get every possible combination of eps and minpts
    db_clus = DBSCAN(eps = ep, min_samples = minpt).fit(auto_sample) # Running DBSCAN while looping through ep and minpt
    if len(np.unique(db_clus.labels_)) == 1: # The next 3 lines is an if statement. By definition, the silhoutte score needs 2 clusters
           sil_score.append(-1) # If there is not 2 unique clusters then the score is technically undefined. This IF STATEMENT takes care
    else: # of that possibility. If there is only 1 unique cluster in db_clus.labels, the we append -1 to sil_score since -1 is the worst
        sil_score.append(silhouette_score(auto_sample, db_clus.labels_))# Otherwise we proceed with calculating the score


# In[23]:


results_dbscan = pd.DataFrame(list(itertools.product(eps,minpts))) # Create dataframe of eps and minpts
results_dbscan['silhouette_score'] = sil_score # Add sil_score to dataframe
results_dbscan.sort_values(by='silhouette_score', ascending=False) # Sort my sil_score


# In[24]:


# The following code will filter the DBSCAN to only the highest silhoutte_score
results_dbscan_filtered = results_dbscan[results_dbscan['silhouette_score'] >= 0.30]
print(results_dbscan_filtered.sort_values(1))


# In[29]:


dbscan_good = DBSCAN(eps = 0.51, min_samples = 3).fit(auto_updated) # picking an eps and min_samples that gives us a very high silhouette_score
auto_updated['DBSCAN_Clusters'] = dbscan_good.labels_ # Append to auto data
print(auto_updated.groupby('DBSCAN_Clusters').describe()) # get summary statistics
print(auto_updated)
print(len(auto_updated[auto_updated.DBSCAN_Clusters == -1]))
print(len(auto_updated[auto_updated.DBSCAN_Clusters == 0]))


# In[27]:


#The fact that the almost all of the dbscan clusters are the same shows that dbscan is not an effective measure for clustering.  


# In[26]:


fig, axes = plt.subplots(2, 2, figsize = (20,20))

ax = sns.violinplot(data = auto_updated, x = "DBSCAN_Clusters", y = "agecat", ax=axes[0,0])
ax.set_xlabel("DBSCAN Clusters", fontsize = 20)
ax.set_ylabel("Age Category",fontsize = 20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax = sns.violinplot(data = auto_updated, x = "DBSCAN_Clusters", y = "credit_score", ax=axes[0,1])
ax.set_xlabel("DBSCAN Clusters",fontsize = 20)
ax.set_ylabel("Credit Score",fontsize = 20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax = sns.violinplot(data = auto_updated, x = "DBSCAN_Clusters", y = "traffic_index", ax=axes[1,0])
ax.set_xlabel("DBSCAN Clusters",fontsize = 20)
ax.set_ylabel("Traffic Index",fontsize = 20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax = sns.violinplot(data = auto_updated, x = "DBSCAN_Clusters", y = "claimcst0", ax=axes[1,1])
ax.set_xlabel("DBSCAN Clusters",fontsize = 20)
ax.set_ylabel("Claim Cost",fontsize = 20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)


# In[30]:


fig, axes = plt.subplots(2, 2, figsize = (20,20))

ax = sns.violinplot(data = auto_updated, x = "DBSCAN_Clusters", y = "veh_age", ax=axes[0,0])
ax.set_xlabel("DBSCAN Clusters", fontsize = 20)
ax.set_ylabel("Vehicle Age",fontsize = 20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax = sns.violinplot(data = auto_updated, x = "DBSCAN_Clusters", y = "veh_value", ax=axes[0,1])
ax.set_xlabel("DBSCAN Clusters",fontsize = 20)
ax.set_ylabel("Vehicle Value",fontsize = 20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax = sns.violinplot(data = auto_updated, x = "DBSCAN_Clusters", y = "numclaims", ax=axes[1,0])
ax.set_xlabel("DBSCAN Clusters",fontsize = 20)
ax.set_ylabel("Number of Claims",fontsize = 20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax = sns.violinplot(data = auto_updated, x = "DBSCAN_Clusters", y = "claimcst0", ax=axes[1,1])
ax.set_xlabel("DBSCAN Clusters",fontsize = 20)
ax.set_ylabel("Claim Cost",fontsize = 20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)


# In[ ]:




