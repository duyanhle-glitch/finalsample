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

1.Profit and Amount by Category line chart

2.Total Quantity Sold by Sub-Category(top 10) bar chart

3.Geographic Analysis: Sales distributions across cities and states to understand regional customer behavior. 

4.Payment Method Quantity distribution(pie chart)

5.Profit by month: Monthly and seasonal sales patterns to inform stock and marketing strategies.(line chart)

## 5.Machine Learning Applications
4.1 Linear Regression for Sales Prediction
To forecast sales, we used Linear Regression with the Order Date and Quantity features to predict Amount.
Evaluation: Mean Absolute Error (MAE) and R-squared metrics showed moderate prediction accuracy.
Visualization: A scatter plot of actual vs. predicted sales highlighted model performance.

4.2 Customer Segmentation with K-Nearest Neighbors (KNN)
For customer segmentation, we applied KNN using features such as Amount, Quantity, and location data (City, State).
Elbow Method: Determined the optimal number of clusters.
PCA Visualization: Showcased customer clusters based on spending patterns and geography.

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




