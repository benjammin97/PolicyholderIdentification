# Policyholder Identification
##### Project Summary
Python code that categorizes auto-insurance customers into groups based on risk level.
##### Skills Used:
###### Python
* Data preprocessing to ensure data quality.
* Implementing multiple unsupervised machine learning models to search for data insights. 
* Performed exploratory data analysis to determine the best fitting clustering model.
* Iterated through several paramaters in each clustering model to optimize each model. 
##### Desired Insights
* How many risk levels are present in the data set?
* What expected cost does each risk level present to the company?

## Preparing the Data
#### [The Dataset](https://github.com/benjammin97/PolicyholderIdentification/blob/main/auto_policies_2020.csv) 
#### Importing the Libraries:
![image](https://user-images.githubusercontent.com/65525140/212584423-cbb7d171-ef71-4f8e-a498-46ccd8bb39f5.png)

#### Importing the Data into Python:
![image](https://user-images.githubusercontent.com/65525140/212584544-a0497d66-5770-4612-8f22-4fe3c6d8fe35.png)

#### Cleaning the Data:
![image](https://user-images.githubusercontent.com/65525140/212584723-ca92618c-0302-4732-a3e7-4ec8d8696716.png)
![image](https://user-images.githubusercontent.com/65525140/212584827-fc2631bd-8ee5-4049-949c-7864529c6977.png)
![image](https://user-images.githubusercontent.com/65525140/212584980-87128cfd-ffa1-4b14-995d-4ef19cddba1e.png)

#### Scaling the Data:
![image](https://user-images.githubusercontent.com/65525140/212585219-d49f1c71-dcdb-4955-9b5a-0f205db47f6e.png)

## Implementing Machine Learning Clustering Models
If you want to see the technical details, you can check out the full code [here](https://github.com/benjammin97/PolicyholderIdentification/blob/main/CategorizingPolicyholders.py).
I had a few goals in mind when I implemented the clustering models:
* Find the optimal clustering model between k-means and dbscan by iterating through all possible paramaters that made logical sense on both models.
* Determine the correct model using the appropriate accuracy metrics.
* Determine the correct number of clusters using the shilouette scores and inertia values.  
  
I was able to do this, and the results were insightful.

## Key Takeaways
* K-means is the most effective clustering model
* There are three primary risk levels for auto-insurance policyholders.

## [Results and Presentation](https://github.com/benjammin97/PolicyholderIdentification/blob/main/Categorizing%20Policyholders%20with%20Unsupervised%20Learning.pptx)

## [Problem Outline](https://github.com/benjammin97/PolicyholderIdentification/blob/main/MSC550%20Fall%202021%20Midterm.pdf)
## [Code](https://github.com/benjammin97/PolicyholderIdentification/blob/main/CategorizingPolicyholders.py)


