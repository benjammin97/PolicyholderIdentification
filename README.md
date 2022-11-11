# Policyholder Identification
Data was preprocessed by eliminating each row with a null value with it(not including zero claims), 
eliminating unneccessary columns, and scaling the data. It reduced the data from 60k rows to 48k rows. 
1/10th of the data was sampled so that each clustering algorithm could be iteratively tested 
using itertool for loops to determine the best fit by using the inertia and silhouette scores. This was done
for lack of computational power on my personal laptop.Once the best algorithm and parameters were 
determined(kmeans, n=3), the algorithm was applied to the full dataset which gave us our results. 
Matplotlib and seaborn visualizations were developed based on the results, 
some of which were developed to show the weight of each factor in determining the categorization.
Number of claims and cost of claims unsurprisingly were the biggest factors for determining
risk level.

## [Problem Outline](https://github.com/benjammin97/PolicyholderIdentification/blob/main/MSC550%20Fall%202021%20Midterm.pdf)
## [Code](https://github.com/benjammin97/PolicyholderIdentification/blob/main/CategorizingPolicyholders.py)
## [Dataset](https://github.com/benjammin97/PolicyholderIdentification/blob/main/auto_policies_2020.csv)
## [Results and Presentation](https://github.com/benjammin97/PolicyholderIdentification/blob/main/Categorizing%20Policyholders%20with%20Unsupervised%20Learning.pptx)
