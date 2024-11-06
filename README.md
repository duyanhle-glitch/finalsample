# ONLINE SALES FINAL REPORT
- **Subject:HSB3119-Introduction to Data Science**
- **Class:MAS02**
- **Lecturer:Dr. Emmanuel Lance Christopher VI M. Plan**
- **Group:06**

## Team Member Evaluation Of Teamwork


## Table of contents
1. [Introduction](#introduction)
    2. [Descriptive statistics](#sec2p2)
    3. [Start looking at categories of diner](#sec2p3)
    4. [Plots to summarize some statistics](#sec2p4)

2. [Data source](#section3)
    1. [Regression in Seaborn](#sec3p1)
    2. [Simple linear regression using polyfit](#sec3p2)
    3. [Regression with statsmodels](#sec3p3)
    4. [Regression with scikit-learn](#sec3p4)
    5. [Linear regression on various subsets of the data](#sec3p5)
    
5. [Relationships between variables](#section4)
    1. [Visualize relationships between numerical variables with pairplot](#sec4p1)
    2. [Investigate relationships between tip amount and the other variables](#sec4p2)
    3. [Does the amount spent depend on party size?](#sec4p3)
    4. [Classification](#sec4p4)
    
6. [Work done by other people on the Tips data set](#section5)
    
7. [Conclusion](#conclusion)

9. [References](#references)


## 1.Introduction  <a name="introduction"></a>

This is a project that aims at looking at the data of an online shop located in India while trying to find out the demographic of the customers, buying behavior, and the sales. The analysis also seeks to determine the factors that contribute towards and affect the sales and retention of clients which will enable the online shops to strategize and increase profits while satisfying their customers.

Our project is centered on two key questions related to online retail operations:

- How can we predict future sales volumes based on past data?
- How can we categorize customers into meaningful segments based on their purchasing behaviors?
  
By answering these questions, we aimed to uncover patterns that allow businesses to anticipate demand and better understand their customers. For the first question, we selected a Linear Regression model to predict future sales values, leveraging features such as previous sales records, product type, and seasonal information. For customer segmentation, we implemented the K-Nearest Neighbors (KNN) algorithm to classify customers into groups based on purchasing history and demographic data.

## 2. Data Sources <a name="section3"></a>

For this particular project the data is sourced from a dataset provided by one Samruddhi Bhosale on Kaggle and through a dataset available on Github. The dataset on Kaggle had these two files at its core: Orders.csv and Details.csv which are now brought together for easier viewing and analysis.

- **Orders.csv**: Contains Order IDs, order dates, customer names, and locations, which give us a snapshot of who the customers are and where they’re from.
- **Details.csv**: Adds specific order details linked to each Order ID, allowing us to understand each purchase’s unique characteristics.

(Exploration: Summarize the data with descriptive statistics (mean, median, standard deviation) to get an overall picture)
First 10 rows of the table:

![image](https://github.com/user-attachments/assets/83f3622a-4176-4274-bb53-581221cb72df)

The table has 1500 rows,11 columns

(Average Amount:291.847333 , Average Profit:24.64200) cai nay optional
details.describe() dung lenh nay
sum of amount,profit,quantity

This project tackles the examination of the data structure and first attempts to assess the student behavior, the level of the product, and what the sales patterns are. The purpose of such activity is to enable the analyst to understand the main client features of the dataset and, in the following phases, prepare the dataset for further analysis.

Dataset Overview
One peculiar facet of the dataset is the following columns which are regarded as able to provide innovative information of clients of diverse purchase elements of interest and their purchase patterns:

- **Order ID**: This is a distinct number that is allocated to every single order so that every order can be traced.

- **Order Date**: Refers to the day the order is placed. It can be useful in determining the trends in sales over time

- **Customer Name, State, City**: Such details help to figure out the region of the customers and assist in targeting the market more effectively or geographically.

- **Amount and Profit**: The amount and profit from each order are shown. These are the most important measures for the assessment of financial results.

- **Quantity**: Is the number of items contained in orders which can assist in large orders or bulk purchasing.

- **Category and sub-category**: type of products so that products may be classified into categories (e.g. Electronics, Furniture) and sub-categories of the same types.

- **Payment Mode**: This shows the type of payment made by the customer (e.g. Credit Card, COD) indicating what payment methods are preferred by the clients.

## 3.Data Preparation and Cleaning 


## 4.Data Exploration
1.Profit by month: Monthly and seasonal sales patterns to inform stock and marketing strategies.
### Monthly Profit Trend Analysis

This line chart displays the **monthly profit trend** over the course of a year. It shows the fluctuations in profit from January to December, with significant variations across different months.

### Interpretation and Key Observations

1. **Steady Decline in Early Months**:
   - From **January to April**, there is a consistent decline in profit, dropping from around $10,000 in January to close to $1,000 in April. This steady downward trend suggests either a seasonal slowdown, reduced sales, or increased costs during these months.
   
2. **Losses in Mid-Year (May)**:
   - In **May**, profit dips below zero, indicating a **net loss**. This may have been caused by factors such as a significant drop in sales, increased operational costs, or discounts and promotions that impacted profit margins.
   - This dip marks the lowest point in the year, with losses close to -$4,000, suggesting this month had particularly challenging financial performance.

3. **Recovery and Growth in Late Year**:
   - From **June to November**, there is noticeable improvement, with profit showing an upward trend despite some fluctuations.
   - **October and November** see a strong spike in profit, with November reaching the highest point after January, nearing around $10,000. This increase could be due to seasonal factors like holiday sales, promotional events, or increased demand towards the end of the year.

4. **End-of-Year Drop (December)**:
   - Profit sharply declines again in **December**. This drop might indicate post-holiday slowdowns, inventory clearance sales at lower profit margins, or other seasonal factors that impact profit.

### Interesting Findings

- **Seasonal Patterns**: There appear to be distinct **seasonal trends** in profit, with early-year (January-April) and late-year (October-November) peaks, and a mid-year dip in May. This pattern could guide planning for inventory, staffing, and promotions.
  
- **Key Focus Months**:
   - **January and November** are peak profit months, indicating high demand or successful sales strategies. These months should be prioritized for marketing and inventory buildup.
   - **May** is a low point, possibly indicating a need to reduce costs or boost sales to avoid future losses. Consider targeted promotions or expense management strategies during this period.

- **End-of-Year Decline**: The drop in December might be typical of post-holiday slowdowns, but monitoring this trend could help adjust end-of-year strategies to mitigate profit loss.

### Actionable Insights
1. **Enhanced Planning for High- and Low-Profit Periods**:
   - Increase marketing and stock for January and November to maximize sales during high-profit months.
   - Consider special promotions or cost-saving measures in May to counteract the low-profit trend.

2. **Seasonal Promotions**:
   - Leverage known high-profit months for launching new products or premium offerings.
   - Use mid-year periods (like May) to introduce discounts or clear excess inventory.

3. **Cost Management in Decline Months**:
   - In months with declining trends (April, May, December), focus on minimizing operational costs or improving efficiency to protect profit margins.


![Screenshot 2024-11-06 222307](https://github.com/user-attachments/assets/f5e7cb56-8d9f-471e-b0f6-9c24a353e9bb)

2.

## 5.Machine Learning Applications
4.1 Linear Regression for Sales Prediction
To forecast sales, we used Linear Regression with the Order Date and Quantity features to predict Amount.
Evaluation: Mean Absolute Error (MAE) and R-squared metrics showed moderate prediction accuracy.
Visualization: A scatter plot of actual vs. predicted sales highlighted model performance.



4.2 Customer Segmentation with K-Nearest Neighbors (KNN)
For customer segmentation, we applied KNN using features such as Amount, Quantity, and location data (City, State).
Elbow Method: Determined the optimal number of clusters.
PCA Visualization: Showcased customer clusters based on spending patterns and geography.

### Interpretation of the Elbow Plot
The **y-axis** represents the **Within-Cluster Sum of Squares (WCSS)**, which is a measure of the variance within each cluster. The **x-axis** represents the number of clusters (`k`).
In this plot:There is a sharp decrease in WCSS from `k=1` to `k=3`, indicating that adding clusters up to this point significantly reduces the within-cluster variance.After **k=3**, the decrease in WCSS becomes more gradual, suggesting that adding more clusters yields diminishing improvements in compactness.

### Optimal Number of Clusters
**k=3** appears to be the "elbow point" in this plot. This is the point where the curve begins to flatten, indicating that three clusters may be optimal, as it balances compactness and interpretability. **k=3** is likely a good choice, as it provides a balance between simplicity and segmentation quality.The elbow plot suggests that **three clusters (k=3)** is an optimal choice for segmenting customers. This choice effectively captures the natural groupings in the data, balancing complexity and accuracy. Moving forward with three clusters allows us to interpret and act on customer segments without introducing unnecessary complexity. This segmentation will support targeted marketing and resource allocation for each group’s unique characteristics.

![Screenshot 2024-11-06 215938](https://github.com/user-attachments/assets/02595688-4f33-4641-82f4-b02c9fcfd188)

2. Silhouette Score Interpretation
Silhouette Score for k=3: 0.53

The silhouette score measures the cohesion and separation of clusters. Scores range from -1 to +1:
+1 indicates well-separated clusters, where points are closer to their own cluster center than to others.
0 indicates overlapping clusters, where points are roughly equidistant between cluster centers.
-1 indicates that points may be in the wrong clusters.
A silhouette score of 0.53 is considered moderate. It suggests that the clusters are reasonably distinct but could be better separated. The score indicates that while the clusters make sense, some overlap exists, or some points are closer to the boundary between clusters.

Evaluation:
Although the silhouette score is not perfect, it is reasonable for a real-world dataset. A score above 0.5 indicates fairly good clustering, but there may be room for improvement.
Silhouette Score: While a silhouette score of 0.53 is acceptable, it suggests that the clusters are not perfectly distinct. In some cases, this could indicate that more distinct clusters might better fit the data, or that the data has some natural overlap between groups.

![Screenshot 2024-11-06 221712](https://github.com/user-attachments/assets/f21b1e35-f5ab-4c47-9431-ca643b9181a1)


### Model Performance Summary:

**Precision, Recall, F1-Score**: The model achieves a perfect score (1.00) for precision, recall, and F1 across all segments. This means that for this specific dataset, the KNN model is performing exceptionally well, correctly classifying each data point into the intended segment.**Accuracy**: The model’s accuracy is 100%, indicating that all predictions match the actual segment labels in the test set.
Precision: Measures the accuracy of the model when it predicts a specific segment. Precision scores are 1.00 for segments 0 and 1, and 0.95 for segment 2, indicating that the model rarely misclassifies these segments.
Recall: Measures the model’s ability to correctly identify all instances of a given segment. Recall scores are 1.00 for segments 0 and 1, and 1.00 for segment 2, meaning that the model identifies nearly all instances in each segment.
F1-Score: The harmonic mean of precision and recall. All segments have high F1-scores (1.00 for segments 0 and 1, and 0.97 for segment 2), indicating balanced precision and recall.
Overall Accuracy: The model achieved an impressive accuracy of 1.00 (100%), which means it correctly classified every instance in the test set.
  
![Screenshot 2024-11-06 220536](https://github.com/user-attachments/assets/8ea8de8b-f72c-4848-8185-cc7e3ee93742)

2. PCA Visualization of Customer Segments
The PCA plot reduces the data to two principal components to visualize the segmentation results, showing clear separation between segments:

Segment 0 (Purple): A compact cluster with high density. This segment likely represents customers with low to medium spending and quantity.

Segment 1 (Blue-Green): Another well-defined cluster, but spread out slightly more than Segment 0, possibly indicating medium spending patterns with a bit more variability in quantity.

Segment 2 (Yellow): Spread over a wider area with a few higher-value outliers. This segment likely represents high-value customers who tend to purchase in larger quantities or have higher transaction amounts.

Insights from the Visualization:
The distinct separation of clusters in the PCA plot shows that our features (Amount, Quantity, and location data) were effective in differentiating customer segments.
Segment 2 appears to have customers who are outliers in terms of spending patterns, which could indicate high-value or high-loyalty customers.

4. Interesting Findings and Business Insights
Distinct Customer Segments:

The model identified three main customer segments with clear separation:
Segment 0 (Low-Value Customers): These customers make low to medium-value purchases with limited quantity. They may be occasional or budget-conscious buyers.
Segment 1 (Medium-Value Customers): Representing the average customer, these buyers make medium-value purchases and could be targeted with upselling and cross-selling strategies.
Segment 2 (High-Value Customers): This segment includes high-spending customers who purchase larger quantities. They likely represent loyal or high-priority customers who could benefit from personalized marketing, loyalty rewards, and exclusive promotions.
Targeted Marketing Strategies:

Segment 0: Consider using discounts or entry-level products to encourage more frequent purchases from this segment. Promoting bundles or budget-friendly options could also attract this group.
Segment 1: This group may be responsive to upselling and cross-selling strategies. Highlight complementary products or related items during checkout to increase their order value.
Segment 2: For high-value customers, consider implementing loyalty programs or personalized recommendations to encourage repeat purchases and reward their loyalty. Exclusive offers or VIP access to new products could be particularly appealing to this segment.
Optimized Resource Allocation:

Inventory and logistics can be optimized based on customer segments. For example, maintaining sufficient stock for products popular among Segment 2 can ensure high-value customers receive priority service, while budget products for Segment 0 can be stocked based on demand patterns.
Summary for the Boss
The KNN model successfully identified three distinct customer segments with high classification accuracy. This segmentation provides valuable insights that can drive targeted marketing, customer retention strategies, and optimized inventory management:



Segment 0 (Low-Spenders): Attract them with discounts or budget-friendly bundles to encourage frequent purchases.
Segment 1 (Average Buyers): Increase their order value with upselling and cross-selling strategies.
Segment 2 (High-Spenders): Retain them with loyalty programs, personalized offers, and exclusive deals.
This segmentation will allow us to cater to each customer group’s specific needs, potentially increasing engagement, revenue, and customer satisfaction across different spending levels.

![Screenshot 2024-11-06 214240](https://github.com/user-attachments/assets/c3db1e97-b221-4577-b665-b5456c1cd9ab)

- **Cluster Segmentation Plot**:
  - The scatter plot shows three distinct clusters (Segments 0, 1, and 2) based on `Quantity` and `Amount`.
  - **Segment 0** (purple): Consists of transactions with lower quantities and lower sales amounts.
  - **Segment 1** (blue-green): Represents medium quantities and sales amounts.
  - **Segment 2** (yellow): Includes transactions with higher amounts and varying quantities, suggesting this may represent high-value purchases or bulk orders.

### Interpretation and Business Implications for the Online Sales Shop:
1. **Identifying Customer Segments**:
   - This segmentation model effectively divides customers into three segments based on purchasing behavior, which can help tailor marketing and sales strategies.
   - **Low-Spending Segment (Segment 0)**: These customers make small, lower-value purchases. They may be occasional buyers or more price-sensitive.
   - **Mid-Spending Segment (Segment 1)**: Customers in this segment make moderate purchases, potentially representing the typical or average buyer.
   - **High-Spending Segment (Segment 2)**: This group consists of high-value transactions, possibly from bulk buyers or loyal customers who make larger purchases.

2. **Targeted Marketing Strategies**:
   - **Segment 0**: Consider promotions or discounts to encourage these customers to increase their order sizes. Incentives like free shipping on minimum orders or bundle deals could be effective.
   - **Segment 1**: Target this segment with upselling or cross-selling strategies, as they are already making mid-range purchases. Highlight complementary products or upgrades.
   - **Segment 2**: High-value customers could be offered loyalty rewards or exclusive discounts to retain them and encourage repeat purchases.

3. **Inventory and Resource Planning**:
   - Knowing the distribution of customer segments can help in planning inventory and staffing based on typical order sizes. For instance, if Segment 2 customers are infrequent but valuable, the shop could focus on keeping items in stock that appeal to this segment without overstocking.

4. **Personalized Customer Experience**:
   - The shop can use this segmentation to personalize the online shopping experience. For example, based on past purchase behaviors, the website could display different product recommendations, highlight popular products for each segment, or adjust the messaging in email campaigns.

### Summary
The KNN model has successfully segmented customers into distinct groups with 100% accuracy. These segments provide actionable insights for personalized marketing, inventory management, and customer retention strategies. Leveraging these insights can help increase engagement, optimize inventory, and ultimately boost sales for the online shop.

![Screenshot 2024-11-06 214610](https://github.com/user-attachments/assets/4e90cb99-37e3-4c74-98cb-c2343bdb9019)







4.3 Logistic Regression for High-Profit Classification
To classify high-profit orders, we used Logistic Regression with Quantity, Category, and Payment Mode as predictors.
Confusion Matrix: Evaluated model accuracy in predicting high vs. low-profit orders.
ROC Curve: Showed model performance in distinguishing between high and low profit


## 6.Conclusions <a name="conclusion"></a>


## References <a name="references"></a>

- [1]  Anaconda Distribution
https://www.anaconda.com/

- [2] Python Software Foundation
https://www.python.org/

- [3] Project Jupyter
https://jupyter.org/




