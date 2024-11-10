# ONLINE SALES FINAL REPORT
- **Subject:HSB3119-Introduction to Data Science**
- **Class:MAS02**
- **Lecturer:Dr. Emmanuel Lance Christopher VI M. Plan**
- **Group:06**

## Team Member Evaluation Of Teamwork


## Table of contents
1. [Introduction](#introduction)

2. [Data source](#section2)
   
    2.1. [Dataset Overview](#sec2p1)
    
3. [Data Preparation and Cleaning](#section3)
 
4. [Charts](#section4)
   
    4.1. [Mothly Profit Trend](#sec4p1)
   
    4.2. [Top 10 most profit product](#sec4p2)
   
    4.3. [Number of orders by category](#sec4p3)
   
    4.4. [Number of orders by payment method](#sec4p4)

5. [Machine Learning Application](#section5)

   5.1.[Linear Regression for Sales Prediction](#sec5p1)

   5.2.[Customer Segmentation with KMeans cluster](#sec5p2)
   
7. [Conclusion](#conclusion)

8. [References](#references)


## 1.Introduction  <a name="introduction"></a>

The purpose of this data report is to analyze sales and customer behavior for an online retail business operating in India. The dataset contains information on transactions, including product categories, payment methods, order quantities, and profit margins. The business aims to understand the key drivers of sales and identify potential segments within its customer base to enhance targeting strategies and optimize inventory management. To achieve this, we explore and analyze various sales metrics and patterns, such as monthly profit trends, order distributions by payment mode, and top-performing products.

We will tackle the problem from both a descriptive and predictive perspective in our analysis. The descriptive analytics will explain what exists regarding sales patterns, the most profitable types of products and categories, while it will also let customers know their preferred payment options for the items they purchase. Visualizations like monthly profit trends and the distribution of orders by category help uncover these trends, allowing the team to spot seasonality effects and popular payment methods. Predictive analytics, including regression modeling, aim to forecast future sales based on historical patterns, supporting inventory planning and demand prediction.

Additionally, we employ customer segmentation using KMeans clustering to classify customers based on purchasing behavior, such as the quantity and value of purchases. This will help in segmenting different customer types and hence come up with specific marketing strategies targeted towards the business. By using the combination of several statistical models and machine learning techniques, this report tries to answer some of the fundamental business questions with respect to sales prediction, customer preference, and product performance to inform data-driven decisions toward improving profitability and enhancing customer satisfaction.

## 2. Data Sources <a name="section2"></a>

For this particular project the data is sourced from a dataset provided by Samruddhi Bhosale on Kaggle and through a dataset available on Github. The dataset on Kaggle had these two files at its core: Orders.csv and Details.csv which are now brought together for easier viewing and analysis.

- **Orders.csv**: Contains Order IDs, order dates, customer names, and locations, which give us a snapshot of who the customers are and where they’re from.
- **Details.csv**: Adds specific order details linked to each Order ID, allowing us to understand each purchase’s unique characteristics.

### 2.1. Dataset Overview <a name="sec2p1"></a>

One peculiar facet of the dataset is the following columns which are regarded as able to provide innovative information of clients of diverse purchase elements of interest and their purchase patterns:

- **Order ID**: This is a distinct number that is allocated to every single order so that every order can be traced.

- **Order Date**: Refers to the day the order is placed. It can be useful in determining the trends in sales over time

- **Customer Name, State, City**: Such details help to figure out the region of the customers and assist in targeting the market more effectively or geographically.

- **Amount and Profit**: The amount and profit from each order are shown. These are the most important measures for the assessment of financial results.

- **Quantity**: Is the number of items contained in orders which can assist in large orders or bulk purchasing.

- **Category and sub-category**: type of products so that products may be classified into categories (e.g. Electronics, Furniture) and sub-categories of the same types.

- **Payment Mode**: This shows the type of payment made by the customer (e.g. Credit Card, COD) indicating what payment methods are preferred by the clients.

![Screenshot 2024-11-09 231317](https://github.com/user-attachments/assets/c6198167-26ad-44af-b241-31176f759f5e)

#### Important Columns for Analysis

- **Quantity**, **Amount** and **Profit**: Critical for understanding financial performance.
- **Category** and **Sub-Category**: Essential for market segmentation analysis.
- **Order Date**: For trend analysis over time.
- **Payment Mode**:For analyzing payment method frequency usage.

## 3.Data Preparation and Cleaning <a name="section3"></a>

The data preparation and cleaning process is crucial in preparing the dataset for analysis and ensuring accurate results. Initially, we performed a thorough check of the data’s structure, data types, and completeness by examining its summary statistics, types, and sample rows. The dataset contains 1,500 entries with 11 columns, including both numerical columns (such as 'Amount,' 'Profit,' and 'Quantity') and categorical columns (like 'CustomerName,' 'State,' 'City,' 'Category,' 'Sub-Category,' and 'PaymentMode'). This initial examination indicated no missing values, making the dataset complete and eliminating the need for imputation.
#### Data Information

```plaintext
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1500 entries, 0 to 1499
Data columns (total 11 columns):
 #   Column        Non-Null Count  Dtype 
---  ------        --------------  ----- 
 0   Order ID      1500 non-null   object
 1   Order Date    1500 non-null   object
 2   CustomerName  1500 non-null   object
 3   State         1500 non-null   object
 4   City          1500 non-null   object
 5   Amount        1500 non-null   int64 
 6   Profit        1500 non-null   int64 
 7   Quantity      1500 non-null   int64 
 8   Category      1500 non-null   object
 9   Sub-Category  1500 non-null   object
 10  PaymentMode   1500 non-null   object
dtypes: int64(3), object(8)
memory usage: 129.0+ KB
```

During exploration, summary statistics highlighted key insights into the data distribution. The 'Amount' and 'Profit' columns had significant variability, with 'Amount' ranging from 4 to 5729 and 'Profit' ranging from -1981 to 1864. This variability suggests a diverse set of transactions, with both highly profitable and unprofitable items. Notably, the mean 'Profit' was 24.64, indicating that while most transactions yielded a profit, some resulted in substantial losses. Furthermore, 'Quantity' had a narrower range, from 1 to 14, with an average of around 3.74 per order. These descriptive statistics helped identify potential outliers in 'Profit' and 'Amount,' which we retained for further analysis as they could represent meaningful high- or low-performing transactions.

#### Data Summary

| Statistic | Amount       | Profit       | Quantity    |
|-----------|--------------|--------------|-------------|
| Count     | 1500.000000  | 1500.00000   | 1500.000000 |
| Mean      | 291.847333   | 24.64200     | 3.743333    |
| Std Dev   | 461.924620   | 168.55881    | 2.184942    |
| Min       | 4.000000     | -1981.00000  | 1.000000    |
| 25%       | 47.750000    | -12.00000    | 2.000000    |
| 50%       | 122.000000   | 8.00000      | 3.000000    |
| 75%       | 326.250000   | 38.00000     | 5.000000    |
| Max       | 5729.000000  | 1864.00000   | 14.000000   |


To make the dataset ready for machine learning, we performed several preprocessing steps. First, date formatting was applied to 'Order Date' by converting it from string format to a datetime format, enabling the extraction of temporal features like 'OrderMonth' and 'OrderDayOfWeek.' These new features capture monthly and weekly patterns in purchasing behavior, which could provide insight into seasonality and day-based trends in sales.

Next, we addressed categorical variables, such as 'Category,' 'Sub-Category,' and 'PaymentMode.' For machine learning, these were converted into numerical representations using one-hot encoding. This transformation enables categorical values to be used in regression and clustering algorithms without introducing unintended ordinal relationships. 

Additionally, the data was checked for duplicates, which were removed to prevent potential biases in analysis. Numerical features were standardized to ensure comparability and to prevent any one feature from disproportionately influencing the model, particularly relevant for 'Amount' and 'Profit' given their wide ranges. The final dataset was verified for consistency and coherence, providing a clean and well-structured base for further exploration and predictive modeling. 

In total, the cleaned dataset consists of 1,500 records, with a total sales 'Amount' of 437,771, a total 'Quantity' of 5,615, and a total 'Profit' of 36,963. These summary values offer a high-level perspective on the data’s scope and scale, setting a foundation for detailed analysis and model development.

#### Totals

- **Total Amount**: 437,771
- **Total Quantity**: 5,615
- **Total Profit**: 36,963

## 4.Charts <a name="section4"></a>
### 4.1.Mothly Profit Trend <a name="sec4p1"></a>

The **Monthly Profit Trend** chart provides valuable insights into the variations in profit over the year. This line chart shows a significant dip in profits around May, indicating a potential seasonal low point in sales or possibly a period of increased returns or operational costs. The profits steadily increase after this dip, with a sharp peak in November, suggesting a strong sales period, possibly due to seasonal events, promotions, or holidays.

Observing these patterns can help the business identify high and low-performing months and plan accordingly. For example, the significant increase in November could be leveraged for additional marketing efforts in the same period next year. Additionally, understanding the low-profit months, such as May, could encourage strategies to boost sales or manage costs better during that period. This trend analysis is an essential tool for strategic planning, allowing the business to align its operational, marketing, and sales efforts with seasonal fluctuations.

![chart1](https://github.com/user-attachments/assets/54ce9459-c59f-425c-b72d-2036d641a84e)

### 4.2.Top 10 most profit product <a name="sec4p2"></a>

The **Top 10 Products by Total Profit** horizontal bar chart provides a clear view of which products contribute the most to the company’s profit. At the top of the list are **Printers** and **Bookcases**, generating the highest profits, followed by **Sarees** and **Accessories**. This insight suggests that items like printers and bookcases might be high-margin products or in high demand, which could drive the majority of the store's profitability.

Interestingly, **Tables** and **Trousers** also contribute significantly, even though they might not immediately be considered top profit-driving products. The inclusion of items like **Phones** and **Hankerchiefs** lower in the top 10 indicates diversity in profitable items, covering both technology and clothing/accessories categories. For strategic decisions, the business might focus on promoting these high-profit products more aggressively or analyzing customer purchase behaviors to further increase sales in these categories. This chart also highlights opportunities to expand inventory in these profitable sub-categories or apply targeted marketing efforts to boost sales further.

![chart3](https://github.com/user-attachments/assets/91a19bbf-ba81-4751-9cd4-3063aea20454)

### 4.3.Number of orders by category <a name="sec4p3"></a>

The **Number of Orders by Category** chart highlights the distribution of orders across major product categories, providing insight into consumer preferences. **Clothing** is the most popular category by a significant margin, accounting for over 900 orders, which is much higher than the other categories. This suggests that clothing products are highly demanded by customers and likely represent a substantial portion of the sales volume.

**Electronics** follows with fewer orders, which may imply that these products are less frequently purchased but could have higher price points or profit margins per unit. **Furniture** has the lowest order count among the three, indicating that these items may be less in demand or are higher-priced, causing customers to purchase them less frequently. 

This distribution allows the business to focus on different strategies for each category. For instance, maximizing inventory and variety within the Clothing category could maintain or boost sales volume. For Electronics and Furniture, a targeted marketing strategy or bundling options could potentially increase their order frequency. Understanding customer demand through such insights helps tailor inventory and promotional efforts to align with what customers value most.

![chart4](https://github.com/user-attachments/assets/cb1b05bc-d710-4608-8e21-6e299376b4e9)

### 4.4.Number of orders by payment method <a name="sec4p4"></a>

The **Number of Orders by Payment Mode** bar chart provides insights into customer preferences for different payment methods. Cash on Delivery (COD) is by far the most popular payment method, with nearly twice as many orders as the next most preferred option, UPI. This preference indicates that customers may feel more secure paying upon delivery rather than upfront. 

UPI follows as the second most popular mode, which suggests that digital payment methods are gaining traction. Debit Card and Credit Card payments are also used but to a lesser extent, while EMI payments are the least common. This information is valuable for the business as it can focus on optimizing the payment experience for the most popular methods, possibly offering additional incentives for digital payments to promote a smoother, cashless transaction process. Additionally, understanding these preferences can help tailor marketing strategies to target customers based on their payment habits, enhancing the shopping experience and potentially boosting conversions.

![chart2](https://github.com/user-attachments/assets/2f1a3c83-27f1-459a-8c4f-5ff430f46bb5)

## 5.Machine Learning Applications <a name="section5"></a>
### 5.1 Linear Regression for Sales Prediction <a name="sec5p1"></a>

Based on the OLS regression results, our model provides valuable insights into the factors influencing sales amount (`Amount`). The **R-squared** value of **0.474** implies that approximately 47.4% of the variability in sales can be explained by the features in the model, such as `Quantity`, `Category`, `Sub-Category`, `PaymentMode`, and the extracted date components (`OrderMonth`, `OrderDayOfWeek`). While this is a moderate R-squared, it suggests that these selected features have a substantial influence on sales, although other unobserved factors may also play a role. The **Adjusted R-squared** of **0.464** is slightly lower, accounting for the complexity of the model by penalizing additional predictors. The minimal difference between R-squared and Adjusted R-squared indicates that the model complexity is not excessive, which is positive for model stability. The **F-statistic** of **46.06** with a highly significant p-value (4.22e-146) confirms that the model, as a whole, significantly improves our understanding of sales variability over a model with no predictors.

Examining the coefficients, we see how different variables impact sales. For example, certain categorical variables (`Category` and `Sub-Category`) and payment modes show significant coefficients, indicating that specific product types or payment methods may contribute positively or negatively to sales. Some of the time-based variables, such as `OrderMonth`, also exhibit strong effects, suggesting possible seasonality in sales. The large confidence intervals for some coefficients (e.g., certain product sub-categories) imply variability in their impact, pointing to potential interaction effects that could be explored further.

```plaintext
    OLS Regression Results                            
==============================================================================
Dep. Variable:                 Amount   R-squared:                       0.474
Model:                            OLS   Adj. R-squared:                  0.464
Method:                 Least Squares   F-statistic:                     46.06
Date:                Sat, 09 Nov 2024   Prob (F-statistic):          4.22e-146
Time:                        22:10:31   Log-Likelihood:                -8707.9
No. Observations:                1200   AIC:                         1.746e+04
Df Residuals:                    1176   BIC:                         1.759e+04
Df Model:                          23                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const        181.7437     15.720     11.561      0.000     150.902     212.586
x1           164.7277     10.199     16.151      0.000     144.717     184.738
x2             7.3860     10.160      0.727      0.467     -12.547      27.319
x3             5.9594     10.226      0.583      0.560     -14.104      26.023
x4           102.2232     44.104      2.318      0.021      15.691     188.755
x5           323.9293     24.672     13.130      0.000     275.524     372.335
x6           169.3071     42.040      4.027      0.000      86.826     251.788
x7           -45.5077     42.166     -1.079      0.281    -128.237      37.221
x8           179.2034     64.403      2.783      0.005      52.846     305.561
x9          -384.6236     43.530     -8.836      0.000    -470.029    -299.218
x10         -132.6832     27.827     -4.768      0.000    -187.280     -78.086
x11         -102.3424     52.177     -1.961      0.050    -204.712       0.028
x12         -123.4917     49.899     -2.475      0.013    -221.392     -25.591
x13          213.4283     62.448      3.418      0.001      90.907     335.950
x14          466.3590     66.254      7.039      0.000     336.370     596.348
x15           64.9673     26.992      2.407      0.016      12.009     117.926
x16         -114.9733     44.027     -2.611      0.009    -201.354     -28.593
x17         -186.4071     45.998     -4.053      0.000    -276.654     -96.160
x18          -99.1426     28.960     -3.423      0.001    -155.962     -42.323
x19         -120.3127     42.019     -2.863      0.004    -202.753     -37.872
x20          584.7535     76.001      7.694      0.000     435.641     733.866
x21          569.9769     54.978     10.367      0.000     462.110     677.844
x22          190.4687     34.081      5.589      0.000     123.601     257.336
x23          -24.4279     31.181     -0.783      0.434     -85.604      36.748
x24          225.1620     39.480      5.703      0.000     147.703     302.621
x25            0.3367     26.040      0.013      0.990     -50.754      51.428
==============================================================================
Omnibus:                     1050.131   Durbin-Watson:                   1.949
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            53993.749
Skew:                           3.775   Prob(JB):                         0.00
Kurtosis:                      34.982   Cond. No.                     1.12e+16
==============================================================================
```

#### Residual Analysis and Interesting Findings

From the **residuals analysis**, we generated two plots to assess model fit and assumptions. The **histogram of residuals** indicates that most residuals are centered around zero, which is ideal. However, there is a slight right skew, with a few outliers on the high side, suggesting some high sales values are not fully captured by the model. The **scatter plot of residuals versus fitted values** reveals that residuals generally stay close to zero across predicted values, though there is a small fan shape, suggesting potential heteroscedasticity (variance changing with fitted values). This could imply that the model might underperform with higher sales amounts or could be improved with additional predictors or transformations.

The **histogram of residuals** visualizes the differences between the actual and predicted sales values. This plot provides a quick check on the residual distribution. Ideally, residuals should follow a normal distribution centered around zero, which would suggest that the model's errors are randomly distributed, indicating a good fit.

In this histogram, the majority of residuals are centered around zero, indicating that the model performs reasonably well for most predictions. However, there is a visible right skew, with some large positive residuals extending past 1000. This skew suggests the presence of **outliers** or cases where the model underestimates the sales amount. The residuals on the far right could represent high-value sales that our model fails to predict accurately. This could be due to factors not captured by the current features or the potential need for additional interaction terms or polynomial terms to account for nonlinear relationships.

These observations suggest that, while the model is fairly accurate, there may be room for improvement, particularly in predicting high-value sales. Addressing these residual outliers might involve refining the feature set or trying different model specifications to better capture the variability in the data. This analysis will help determine areas to focus on for model enhancement, particularly to improve predictions for high-sales cases.

![linear1](https://github.com/user-attachments/assets/437ab55b-fd34-4833-9bae-00cc44400bec)


The **Residuals vs Fitted Values** plot provides insight into the model's accuracy and any potential patterns or issues in the predictions. In an ideal regression model, residuals should be randomly scattered around the horizontal axis (at zero) with no clear patterns, indicating that the model captures the relationship between the predictors and the target variable effectively. 

In this plot, most residuals are relatively close to zero for lower fitted values, indicating that the model predicts well for lower sales amounts. However, as the fitted values increase, the residuals begin to show greater variance, with some significant positive outliers for predictions above 1000. This pattern suggests that the model may struggle to predict high sales amounts accurately, possibly due to unaccounted variability or nonlinear relationships not captured by the linear model. 

These findings suggest that while the model is reasonably accurate for lower to mid-range sales predictions, there is room for improvement in predicting higher sales amounts. Enhancements could include exploring additional features, nonlinear transformations, or even alternative models that better handle the high variance observed in the predictions for larger values.

![linearchart2](https://github.com/user-attachments/assets/d92715b1-2c95-4aa2-9f12-b4c9df4466db)


### 5.2 Customer Segmentation with KMeans cluster <a name="sec5p2"></a>

The Elbow Method plot helps us determine the optimal number of clusters for customer segmentation. In this graph, the "Within-Cluster Sum of Squares" (WCSS) is plotted against different values of `k` (number of clusters). As `k` increases, the WCSS decreases, indicating that clusters are becoming more compact. However, the rate of decrease starts to slow around `k=3`, forming an "elbow" shape. This suggests that adding more clusters beyond `k=3` does not significantly improve compactness and could lead to diminishing returns. 

Therefore, `k=3` is a suitable choice for our segmentation, as it balances capturing meaningful clusters with minimizing within-cluster variance. This optimal clustering divides customers into three segments, each with distinct purchasing behaviors, which can then be targeted with tailored marketing strategies. This analysis, combined with the silhouette score and visual segmentation, enables us to better understand and engage with our customer base in ways that enhance profitability and customer satisfaction.

![kmeans1](https://github.com/user-attachments/assets/648048a4-d56b-49d4-8fc8-cee040e30afd)


The customer segmentation analysis reveals insightful patterns in purchasing behavior, based on the clustering of customers into distinct segments. Using the KMeans clustering algorithm on features like `Amount` and `Quantity`, we identified three main customer groups. Segment 0, marked in purple on the visualization, includes customers who tend to have lower spending amounts but a broad range of purchased quantities. This segment might represent customers who make frequent but lower-value purchases, possibly looking for smaller, cost-effective items. Segment 1, shown in teal, represents customers with both low spending amounts and low quantities, possibly indicating single-item purchases or low-budget transactions. This group could consist of casual buyers or those who occasionally make minimal purchases.

In contrast, Segment 2, highlighted in yellow, stands out with higher spending values, suggesting a group of high-value customers who invest significantly in their purchases. The variety in the quantity range within this segment implies that these are possibly loyal customers who make more substantial purchases. Segment 2 could be particularly valuable to the business due to their higher average spend, making them ideal candidates for targeted loyalty programs or premium offerings.

One interesting observation from this clustering is the potential to develop targeted marketing strategies. By understanding that Segment 0 and Segment 1 may consist of budget-conscious or casual buyers, the business could consider offering these segments promotions or discounts to encourage higher spending. Conversely, high-spend customers in Segment 2 could be engaged with exclusive offers or rewards, building loyalty and enhancing their customer lifetime value. The silhouette score of 0.53 indicates moderate clustering quality, suggesting there’s reasonable separation among these segments but also room for refinement. By experimenting with additional features, such as profit margins or seasonal purchase data, the business could further enhance segmentation, enabling even more precise and effective strategies to engage with each customer type. This segmentation sheds light on diverse customer behaviors, allowing for a data-driven approach to relationship-building and profit maximization.

![kemans1](https://github.com/user-attachments/assets/d19db4b0-a929-46b2-8d33-a9a7e2619cc9)

## 6.Conclusions <a name="conclusion"></a>

In conclusion, this analysis provided valuable insights into the sales dynamics and customer behavior for an online retail business. By exploring and visualizing data on monthly profit trends, product profitability, order distribution by category, and payment preferences, we identified key factors influencing sales and customer choices. Predictive modeling using linear regression allowed us to anticipate future sales trends, with significant predictors such as product category, sub-category, and payment method revealing actionable insights. The residual analysis highlighted areas where model accuracy could be improved, especially for higher sales amounts, suggesting the need for further refinement or additional features. Customer segmentation through KMeans clustering effectively categorized customers into three distinct segments, each with unique spending and purchasing behaviors. This segmentation not only highlighted high-value customers but also uncovered potential strategies for engaging budget-conscious buyers. Supported by data-driven insights, these findings empower the business to make informed decisions around product marketing, inventory management, and targeted promotions, ultimately enhancing profitability and customer satisfaction. The analysis successfully answered our initial questions by identifying sales drivers and customer segments, laying a foundation for strategic growth.

## References <a name="references"></a>

- [1]  Anaconda Distribution
https://www.anaconda.com/

- [2] Python Software Foundation
https://www.python.org/

- [3] Project Jupyter
https://jupyter.org/




