# ONLINE SALES FINAL REPORT
- **Subject:HSB3119-Introduction to Data Science**
- **Class:MAS02**
- **Lecturer:Dr. Emmanuel Lance Christopher VI M. Plan**
- **Group:06**

## Team Member Evaluation Of Teamwork
#### Team leader:Lê Duy Anh
(demo)
![Screenshot 2024-11-04 214046](https://github.com/user-attachments/assets/4a3958b0-8ff8-4a17-b5c2-17b65b3f865c)

## I.Introduction

This is a project that aims at looking at the data of an online shop located in India while trying to find out the demographic of the customers, buying behavior, and the sales. The analysis also seeks to determine the factors that contribute towards and affect the sales and retention of clients which will enable the online shops to strategize and increase profits while satisfying their customers.

Our project is centered on two key questions related to online retail operations:

- How can we predict future sales volumes based on past data?
- How can we categorize customers into meaningful segments based on their purchasing behaviors?
  
By answering these questions, we aimed to uncover patterns that allow businesses to anticipate demand and better understand their customers. For the first question, we selected a Linear Regression model to predict future sales values, leveraging features such as previous sales records, product type, and seasonal information. For customer segmentation, we implemented the K-Nearest Neighbors (KNN) algorithm to classify customers into groups based on purchasing history and demographic data.

#### Data Sources

For this particular project the data is sourced from a dataset provided by one Samruddhi Bhosale on Kaggle and through a dataset available on Github. The dataset on Kaggle had these two files at its core: Orders.csv and Details.csv which are now brought together for easier viewing and analysis.

- **Orders.csv**: Contains Order IDs, order dates, customer names, and locations, which give us a snapshot of who the customers are and where they’re from.
- **Details.csv**: Adds specific order details linked to each Order ID, allowing us to understand each purchase’s unique characteristics.
  
To get the data ready for analysis, we cleaned it by addressing missing values, duplicates, and other inconsistencies and luckily we found no missing and duplicate values in the dataset, so no imputation was necessary.These steps were essential to make sure our insights are based on accurate and reliable data.

---

## II.Data Discussion
(Exploration: Summarize the data with descriptive statistics (mean, median, standard deviation) to get an overall picture)
First 10 rows of the table:

![image](https://github.com/user-attachments/assets/83f3622a-4176-4274-bb53-581221cb72df)

The table has 1500 rows,11 columns

(Average Amount:291.847333 , Average Profit:24.64200) cai nay optional
details.describe() dung lenh nay

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

## III.Charts


1.Total Quantity Sold cai nay bar chart by sub 10 cai

2.Profit, and Amount cai nay subplot 2 line graph lam 1 by sub 10 cai

(optional:top theo city hoac state,customer name)

3.

4.

## IV.Machine Learning Applications
1.Linear Regression model to predict future sales values,

2.For customer segmentation,K-Nearest Neighbors (KNN) algorithm to classify customers into groups based on purchasing history and city-sate data.


## V.Conclusions


## References

- https://www.kaggle.com/datasets/samruddhi4040/online-sales-data




