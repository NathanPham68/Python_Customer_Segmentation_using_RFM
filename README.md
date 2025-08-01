# 🛒 [Python] How Can We Segment Customers to Improve Targeted Marketing? - Retail Analytics

<img width="1121" height="643" alt="image" src="https://github.com/user-attachments/assets/85915f37-c102-44a0-99e1-fb6c598b5d56" />

## 📑 Table of Contents

[📌 1. Project Overview](#project-overview)  

[🎯 2. Objective](#objective)  

[🧠 3. Business Context](#business-context)  

[📈 4. Who Is This Project For?](#Who-Is-This-Project-For?)  

[🔎 5. RFM Analysis Overview](#RFM-Analysis-Overview)

[🔍 6. Dataset Description](#Dataset-Description)  

[🔧 7. Methodology](#Methodology)

[📈 8. Analysis & Results](#Analysis-&-Results)

[🧮 9. Key Takeaways](#Key-Takeaways)

## 📌 Project Overview

This project focuses on performing customer segmentation using RFM analysis (Recency – Frequency – Monetary) for SuperStore, a global retail company. The main goal is to support the Marketing Department in designing personalized campaigns for different customer groups during the Christmas and New Year season.

In previous years, the marketing team segmented customers manually using Excel. However, with the increasing size of the customer base and sales data, they needed the Data Analysis Department to build an automated, scalable solution using Python and data science techniques.

This project involves data preparation, RFM score calculation, segmentation, visualization, and providing actionable recommendations for the Marketing and Sales teams to optimize their strategies.

## 🎯 Objective

✔️ Apply RFM analysis to evaluate customer value.

✔️ Segment customers based on their RFM scores.

✔️ Support marketing and sales teams in planning targeted and effective promotional strategies.

## 🧠 Business Context

SuperStore is a global online retailer specializing in unique gift products. With a growing number of customers, especially wholesalers, the company aims to reward loyal customers and re-engage potentially valuable ones.

To do so, the Marketing team requested a customer segmentation model that groups customers based on their behavior throughout the year 2011. The segmentation results will be used to:

* Appreciate loyal customers

* Identify promising customers for upselling

* Re-engage inactive or at-risk customers

## 📈 Who Is This Project For?

✔️ Marketing Teams – To design personalized offers for each customer segment.

✔️ Business Analysts & Data Scientists – To build and apply data-driven customer segmentation.

✔️ Executives – To make informed decisions on customer engagement and lifetime value optimization.

## 🔎 RFM Analysis Overview

RFM (Recency, Frequency, Monetary) is a customer analysis technique based on purchasing behavior. In RFM analysis, each customer is assigned a score based on these three factors. The data is then used to categorize customers into segments, helping businesses identify key audiences for targeted marketing and sales strategies.

* Recency: Measures the time elapsed since a customer's last purchase.
* Frequency: Evaluates how often a customer makes transactions.
* Monetary: Calculates the total amount spent by the customer.

By applying RFM, businesses can segment customers based on their value, allowing them to optimize marketing and customer engagement strategies.

## 🔍 Dataset Description

[**Link to dataset**](https://drive.google.com/drive/folders/1-Dhrd-__D244PINUga9lLeXVZHJDO-iK?usp=sharing)

This dataset contains transaction records between 01/12/2010 and 09/12/2011 from a UK-based online store.

<details>
  <summary><strong>Sheet 1: E-commerce Retail</strong></summary>

Sheet 1 (541,910 rows × 8 columns) contains transaction-level data, including order details, customer IDs, and purchase information.

| Column Name  | Data Type         | Description  |  
|-------------|-----------------|--------------|  
| **InvoiceNo**  | `object`  | Unique invoice number for each transaction (6-digit). If it starts with 'C', it indicates a cancellation. |  
| **StockCode**  | `object`  | Unique product (item) code (5-digit). |  
| **Description**  | `object`  | Product (item) name. |  
| **Quantity**  | `int64`  | The number of units purchased per transaction. |  
| **InvoiceDate**  | `datetime64[ns]`  | Date and time when the transaction occurred. |  
| **UnitPrice**  | `float64`  | Price per unit of the product in sterling. |  
| **CustomerID**  | `float64`  | Unique 5-digit identifier for each customer. |  
| **Country**  | `object`  | Name of the country where the customer resides. |  

</details>

<details>
  <summary><strong>Sheet 2: Segment Mapping</strong></summary>

Sheet 2 stores customer segments along with their RFM scores.

| Segment            | RFM Scores |
| ------------------ | ------------------ |
| Champions          | 555, 554, 544, 545, 454, 455, 445 |
| Loyal              | 543, 444, 435, 355, 354, 345, 344, 335 |
| Potential Loyalist | 553, 551, 552, 541, 542, 533, 532, 531, 452, 451, 442, 441, 431, 453, 433, 432, 423, 353, 352, 351, 342, 341, 333, 323 |
| New Customers      | 512, 511, 422, 421, 412, 411, 311 |
| Promising          | 525, 524, 523, 522, 521, 515, 514, 513, 425,424, 413,414,415, 315, 314, 313 |
| Need Attention     | 535, 534, 443, 434, 343, 334, 325, 324 |
| About to Sleep     | 331, 321, 312, 221, 213, 231, 241, 251 |
| At Risk            | 255, 254, 245, 244, 253, 252, 243, 242, 235, 234, 225, 224, 153, 152, 145, 143, 142, 135, 134, 133, 125, 124 |
| Cannot Lose Them   | 155, 154, 144, 214,215,115, 114, 113 |
| Hibernating        | 332, 322, 233, 232, 223, 222, 132, 123, 122, 212, 211 |
| Lost Customers     | 111, 112, 121, 131,141,151 |

</details>

## 🔧 Methodology
<details>
<summary> <strong>Step 1: Data Preparation</strong></summary>

- Filter and clean the transaction data

- Remove canceled and null entries

- Group data by customer for analysis

</details>

<details>
<summary> <strong>Step 2: Calculate RFM Metrics</strong></summary>

- Recency (R): Days since the last purchase (reference date: 31/12/2011)

- Frequency (F): Total number of purchases

- Monetary (M): Total revenue generated by each customer

</details>

<details>
<summary> <strong>Step 3: Scoring RFM</strong></summary>

- Each metric is scored on a scale of 1 to 5, using quintile ranking

- Higher scores indicate more favorable behavior

</details>

<details>
<summary> <strong>Step 4: Segment Customers</strong></summary>

- Combine RFM scores into a 3-digit code (e.g., R=5, F=5, M=5 → 555)

- Use predefined rules to assign customers into segments (e.g., "Champions", "Loyal", "At Risk", etc.)

</details>

<details>
<summary> <strong>Step 5: Visualize and Analyze</strong></summary>

- Analyze the distribution of customers across segments

- Create visual dashboards to support interpretation

</details>

<details>
<summary> <strong>Step 6: Business Recommendations</strong></summary>

- Suggest which RFM dimension (R, F, or M) should be prioritized based on SuperStore's retail model

- Provide actionable insights to the Marketing and Sales departments

</details>

## 📈 Analysis & Results

[**Link to code**](https://colab.research.google.com/drive/1XYqTkaKBxrIhCmbzMfv44dSsYL2dLOzh?usp=sharing)

### 🧹 I. EDA

#### ✅ 1. Import Packages

```ruby
!pip install pandas==2.1.4 ydata-profiling==4.6.4
!pip install squarify
!pip install pydantic-settings
```

```ruby
import pandas as pd
from ydata_profiling import ProfileReport
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
import squarify    # algorithm for treemap
```

#### ✅ 2. Understand about the data

##### 2.1. Load Dataset

```ruby
path = '/content/drive/MyDrive/DAC 1 on 1 /Python/Project_3_ecommerce_retail/Dataset'

import os
os.chdir(path)

df = pd.read_excel('ecommerce retail.xlsx', sheet_name = 'ecommerce retail')
df.head()
```

<img width="1121" height="277" alt="image" src="https://github.com/user-attachments/assets/0a63739a-7230-441c-8c69-97ade8cfb71f" />

##### 2.2. Get infor about data type & data value

<img width="1121" height="430" alt="image" src="https://github.com/user-attachments/assets/6b1a484d-7230-4bbc-bcfd-b4ee18772429" />

<img width="1121" height="580" alt="image" src="https://github.com/user-attachments/assets/627b0a55-168a-4493-8c14-d132f2f278f2" />

⚡ During the initial data exploration, it was observed that the Quantity and Unit Price columns contain negative values. This is not logically valid and requires further investigation. Possible actions include:

- Checking for data entry errors.

- Identifying whether negative values indicate refunds or cancellations.

- Removing or adjusting incorrect data points to ensure accuracy in analysis.

⚡ Manual Review Required

Some orders contain incorrect descriptions → Perform a manual check and mark them as errors to facilitate further processing.

⚡ Data Validation & Error Identification

* Flagged Erroneous Orders: Merged description_check_update with ecommerce to detect inconsistencies in product descriptions.

* Checked Negative Quantities & Cancellations: Identified if negative Quantity values correspond to cancellations by verifying InvoiceNo starting with "C".

-> Review flagged errors and cancellations to ensure data integrity before analysis.

🚨 Identifying Invalid Transactions

After flagging cancellations, verify if there are any transactions where Quantity is negative, but the InvoiceNo does not start with "C".

These may indicate data inconsistencies.

##### 2.3. Using ProfileReport to Understand more about Category Data Type

```ruby
# profile giúp detect sâu hơn (các cột mang tính chất category, vân vân)
profile = ProfileReport(df)
profile
```

<img width="1121" height="642" alt="image" src="https://github.com/user-attachments/assets/37870be1-5317-4d13-afab-66c137fd406b" />

##### 2.4. Identify the Reason for Unreasonable Data Values (Quantity < 0)

<img width="1121" height="386" alt="image" src="https://github.com/user-attachments/assets/51a21006-3a52-40c6-ac8d-54901be10d54" />

<img width="1121" height="455" alt="image" src="https://github.com/user-attachments/assets/923bce07-83b8-41d1-8149-2bb5ec8d136f" />

<img width="1121" height="287" alt="image" src="https://github.com/user-attachments/assets/aa796941-a1b6-4b29-a8b1-a98fcb087e7d" />

##### 2.5. Identify the Reason for Unreasonable Data Values (Price < 0)

<img width="1121" height="286" alt="image" src="https://github.com/user-attachments/assets/fb0a9a8b-7a50-4b4c-8673-3aff3c63b00f" />

##### 2.6. Handling Invalid Data Types and Data Values

<img width="1121" height="511" alt="image" src="https://github.com/user-attachments/assets/d3790d53-623f-4e94-a11f-d6e5d432d741" />

<img width="1121" height="697" alt="image" src="https://github.com/user-attachments/assets/54fa9a7a-12c3-4339-adf2-619b184183fd" />

#### ✅ 3. Handling Missing Values and Duplicates

##### 3.1. Summary of Columns with Missing Values

```ruby
print('Thống kê những cột có missing value')
print('')
missing_dict = {
                'volume': df.isnull().sum(),
                'percent': df.isnull().sum() / (df.shape[0])}

missing_df = pd.DataFrame.from_dict(missing_dict)
missing_df
```

<img width="1121" height="406" alt="image" src="https://github.com/user-attachments/assets/2a7b9408-9a41-438c-a456-a3d084ada6dd" />

So around **25%** of records don't have customer id value, such records are not useful for the RFM analysis. But let's check further if there are any common records which have null and non-null customer ID but same invoice number, so that we can fill the records with same customer ID and try to decrease the loss.

<img width="1121" height="179" alt="image" src="https://github.com/user-attachments/assets/3e46dd04-8b53-429f-bad1-b49b037a52f8" />

Here we found that the count of the Invoice with null customerID is equivalent to the number of records with missing CustomerID (**132220**). Therefore we are unable to prevent the loss and have to remove all such records before any further analysis.

##### 3.2. Investigating the Reasons Behind High Missing Data

<img width="1121" height="598" alt="image" src="https://github.com/user-attachments/assets/4510d35a-3643-42b7-8750-914e55aed674" />

<img width="1121" height="667" alt="image" src="https://github.com/user-attachments/assets/efccebb3-bb64-4787-bc9a-3a07431c4cbb" />

🧐 Key Findings:

* Missing CustomerID values are spread across all months & countries, not just a specific region or time period.

* Likely due to human errors (e.g., incomplete updates) or system recording issues.

🛠 Solution: Since CustomerID is essential, drop missing values to maintain data integrity.

##### 3.3. Handling Missing Values

<img width="1121" height="326" alt="image" src="https://github.com/user-attachments/assets/57436e31-713f-403f-99fa-0a7669a769cd" />

##### 3.4. Check duplicate value in dataset

<img width="1121" height="296" alt="image" src="https://github.com/user-attachments/assets/444d21b9-db1c-47f3-8776-ef34471f35ca" />

##### 3.5. Reasons for Duplicate Records

<img width="1121" height="539" alt="image" src="https://github.com/user-attachments/assets/2f45ede9-4ecc-44ec-af80-f2a5766a0de4" />

##### 3.6. Handling Duplicate Records

<img width="1121" height="445" alt="image" src="https://github.com/user-attachments/assets/5d7dd163-fdfb-4a4c-8e85-2ee11828093e" />

<img width="1121" height="248" alt="image" src="https://github.com/user-attachments/assets/13a8dd1a-b0d7-44d9-bfac-599d6bcc2a59" />

### 🛠 II. Data Processing

RFM analysis is a customer segmentation technique that uses past purchase behavior to divide customers into groups. RFM helps divide customers into various categories or clusters to identify customers who are more likely to respond to promotions and also for future personalization services.

RFM Attribute Creation(Feature Engineering)

*   R (Recency): Number of days since last purchase

*   F (Frequency): Number of tracsactions

*   M (Monetary): Total amount of transactions (revenue contributed)

#### ✅ 1. RFM Variables

<img width="1121" height="662" alt="image" src="https://github.com/user-attachments/assets/40f69cd8-8fde-44e2-a03f-6513ae0c33e7" />

<img width="1121" height="363" alt="image" src="https://github.com/user-attachments/assets/8c12cfc7-9692-497e-a5df-c883d5915509" />

<img width="1121" height="713" alt="image" src="https://github.com/user-attachments/assets/dda784ed-bbe4-46e8-aa04-6c31a220a24c" />

#### ✅ 2. Loyalty Variables: Loyal and Non-Loyal Customers & Key Customer Characteristics

<img width="1121" height="572" alt="image" src="https://github.com/user-attachments/assets/7f27335d-28ee-4cd6-ad26-787f50a19216" />

<img width="1121" height="389" alt="image" src="https://github.com/user-attachments/assets/738eba87-b08b-4b8b-8e2d-88a23c8f5d12" />

<img width="1121" height="596" alt="image" src="https://github.com/user-attachments/assets/0b7c0927-064b-498b-97ee-c7bf705565f8" />

<img width="1121" height="501" alt="image" src="https://github.com/user-attachments/assets/04c1df72-6ca9-432c-8136-dc61048850f5" />

### 📊 III. Visualization

#### ✅ 1. Using RFM model to launch Marketing campaign to thank customers

```ruby
# In which month have we gained the highest sales?
# Let's visualize the top grossing months

monthly_gross = df_drop_duplications[df_drop_duplications.InvoiceDate.dt.year==2011].groupby(df_drop_duplications.InvoiceDate.dt.month).cost.sum()
plt.figure(figsize=(10,5))
sns.lineplot(y=monthly_gross.values, x=monthly_gross.index, marker='o');
plt.xticks(range(1,13))
plt.show()
```

<img width="1121" height="613" alt="image" src="https://github.com/user-attachments/assets/8d22267f-9a1e-4f19-8073-a6ecde480f7f" />

An increasing pattern can be observed month by month wise with a sharp decline in the month of December.

```ruby
# Quy mô từng Segment theo số lượng khách hàng
segments = RFM_df_final['Segment'].value_counts()

labels = [f"{seg}\n({count} customers)" for seg, count in zip(segments.index, segments.values)]

# color=['grey','orange','pink','purple', 'brown', 'blue', 'green', 'red', 'yellow', 'gray', 'white']
color=sns.color_palette("Set3", len(segments))

fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(14, 8)

squarify.plot(sizes= segments.values,
              label=labels,
              color = color,
              text_kwargs={'fontsize': 10, 'fontweight': 'regular', 'color': 'black'},
              alpha=0.6,)
plt.title("Customer Distribution by RFM Segment (Treemap)",fontsize=18,fontweight="bold")
plt.axis('off')
plt.show()
```

<img width="1121" height="707" alt="image" src="https://github.com/user-attachments/assets/e4769deb-8cac-4036-a0e0-4691273e5f57" />

```ruby
# Distribution (số lượng Customer, doanh thu) theo các Segment. Từ đó detect nhóm Khách hàng đa số, và có các chiến lược Marketing phù hợp

# Set the figure size
segment_by_user_count = RFM_df_final[['Segment','CustomerID']].groupby(['Segment']).count().reset_index().rename(columns = {'CustomerID':'user_volume'})
segment_by_user_count['contribution_percent'] = round(segment_by_user_count['user_volume'] / segment_by_user_count['user_volume'].sum() * 100)
segment_by_user_count['type'] = 'user contribution'

segment_by_spending = RFM_df_final[['Segment','Monetary']].groupby(['Segment']).sum().reset_index().rename(columns = {'Monetary':'spending'})
segment_by_spending['contribution_percent'] = segment_by_spending['spending'] / segment_by_spending['spending'].sum() * 100
segment_by_spending['type'] = 'spending contribution'

segment_agg = pd.concat([segment_by_user_count, segment_by_spending])

plt.figure(figsize=(13, 8))
sns.barplot(segment_agg, x="Segment", y="contribution_percent", hue="type")
plt.title('The overal distribution of User Profile using RFM Model')
plt.xticks(fontsize=6)

# Show the plot
plt.show()
```

<img width="1121" height="684" alt="image" src="https://github.com/user-attachments/assets/666f3479-e0ca-4b6c-823a-5b43f0f94f29" />

📌 **Definition**

* Segment: RFM

* user contribution: Percentage of users (with spending) that fall into each RFM segment

<img width="825" height="120" alt="image" src="https://github.com/user-attachments/assets/7165f350-9594-4678-ab82-a576c3537b56" />

* spending contribution: Percentage of total spending (from users with spending) contributed by each RFM segment

<img width="825" height="126" alt="image" src="https://github.com/user-attachments/assets/6e9ed3fb-0d5b-4ab6-9d66-7cf4ce8b33bc" />

💡 **Insights & Action**

*1. “At Risk” and “Cannot Lose Them” Customers*

These are two critical customer segments, as they represent a large portion of the user base and contribute significantly to total revenue.

However, their high proportion is also a warning sign, since these customers have not engaged with the product for a while and are at high risk of churn.

→ Recommended Actions: 

* Launch re-engagement campaigns with attractive promotions

* Send personalized notifications or reminders to encourage product usage

* Offer limited-time incentives to draw them back

*2. “Loyal”, “New Customers”, “Potential Loyalists” and “Promising” Segments*

These segments represent the majority of customers, but their average transaction value is relatively low, meaning their contribution to total revenue is still limited.

→ Recommended Actions:

* Implement cross-selling and upselling campaigns

* Encourage higher spending through bundles, loyalty points, or product recommendations

* Design personalized offers to increase purchase frequency and order value

#### ✅ 2. Exploit potential customers to become loyal customers

```ruby
# Function to calculate percentile thresholds
def percentile_threshold(column):
    p_25 = column.quantile(0.25)
    p_50 = column.quantile(0.5)
    p_75 = column.quantile(0.75)
    quantile_list = [p_25, p_50, p_75]
    return quantile_list

# Function to assign quantile thresholds
def quantile_threshold(x, quantile_list):
    if x <= quantile_list[0]:
        value = f'q1: < {round(quantile_list[0])}'  # Format as string with 2 decimal places
    elif x <= quantile_list[1]:
        value = f'q2: {round(quantile_list[0])} - {round(quantile_list[1])}'
    elif x <= quantile_list[2]:
        value = f'q3: {round(quantile_list[1])} - {round(quantile_list[2])}'
    else:
        value = f'q4: > {round(quantile_list[2])}'
    return value

# Apply thresholds and assign bins for Quantity average
quantity_average_list = percentile_threshold(RFM_df_final['Quantity_average'])
RFM_df_final['Quantity_average_bin'] = RFM_df_final['Quantity_average'].apply(
    lambda x: quantile_threshold(x, quantity_average_list)
)

# Apply thresholds and assign bins for Cost average
cost_average_list = percentile_threshold(RFM_df_final['Cost_average'])
RFM_df_final['Cost_average_bin'] = RFM_df_final['Cost_average'].apply(
    lambda x: quantile_threshold(x, cost_average_list)
)

# Apply thresholds and assign bins for First Quantity
first_quantity_list = percentile_threshold(RFM_df_final['First Quantiy'])
RFM_df_final['First Quantity bin'] = RFM_df_final['First Quantiy'].apply(
    lambda x: quantile_threshold(x, first_quantity_list)
)

# Apply thresholds and assign bins for First Cost
first_cost_list = percentile_threshold(RFM_df_final['First cost'])
RFM_df_final['First cost bin'] = RFM_df_final['First cost'].apply(
    lambda x: quantile_threshold(x, first_cost_list)
)

# Display the updated DataFrame
RFM_df_final.head()
```

<img width="1121" height="495" alt="image" src="https://github.com/user-attachments/assets/d12421b5-acde-425d-afea-6db2a3b44a48" />

```ruby
# Ensure loyal_customer_volume and non_loyal_customer_volume are calculated
loyal_customer_volume = RFM_df_final[RFM_df_final['loyal_status'] == 'Loyal']['CustomerID'].nunique()
non_loyal_customer_volume = RFM_df_final[RFM_df_final['loyal_status'] == 'Non Loyal']['CustomerID'].nunique()

# Group by Quantity_average_bin and count CustomerID
df = RFM_df_final[['CustomerID', 'Quantity_average_bin', 'loyal_status']].groupby(['Quantity_average_bin', 'loyal_status']).count().reset_index().rename(columns={'CustomerID': 'Customer_volume'})

# Calculate Customer_percent
df['Customer_percent'] = df.apply(
    lambda x: (x.Customer_volume / loyal_customer_volume) * 100 if x.loyal_status == 'Loyal' else (x.Customer_volume / non_loyal_customer_volume) * 100,
    axis=1
)

# Drop Customer_volume column
df = df.drop(columns=['Customer_volume'])

# Pivot table to prepare for plotting
df_pivot = df.pivot(index='loyal_status', values='Customer_percent', columns='Quantity_average_bin')

# Plot the 100% stacked bar chart
df_pivot.plot(kind='bar', stacked=True, figsize=(10, 6), title = 'Stacked Bar Chart Illustrating the Distribution of Customers by Loyalty Status and Average Purchase Quantity')
```

<img width="1121" height="666" alt="image" src="https://github.com/user-attachments/assets/6300ad26-15f4-495e-a258-8c9832a6946e" />

📌 **Definition**

Quantity_Average_Bin: Average Purchase Quantity Thresholds

Example: q1 < 6 -> This group includes customers who, on average, purchase fewer than 6 items per transaction.

💡 **Insights & Action**

Loyal customers tend to make frequent purchases, but with a moderate quantity per order (typically fewer than 15 items per transaction)

This pattern suggests they are likely individual consumers rather than bulk buyers — yet they demonstrate strong, long-term engagement and value.

-> Actions:

* Target this customer segment with personalized marketing campaigns

* Use loyalty programs, email marketing, and retargeting ads to retain and deepen engagement

* Highlight value-added services (e.g., fast delivery, exclusive deals) to strengthen brand loyalty

```ruby
# Ensure loyal_customer_volume and non_loyal_customer_volume are calculated
loyal_customer_volume = RFM_df_final[RFM_df_final['loyal_status'] == 'Loyal']['CustomerID'].nunique()
non_loyal_customer_volume = RFM_df_final[RFM_df_final['loyal_status'] == 'Non Loyal']['CustomerID'].nunique()

# Group by Quantity_average_bin and count CustomerID
df = RFM_df_final[['CustomerID', 'Cost_average_bin', 'loyal_status']].groupby(['Cost_average_bin', 'loyal_status']).count().reset_index().rename(columns={'CustomerID': 'Customer_volume'})

# Calculate Customer_percent
df['Customer_percent'] = df.apply(
    lambda x: (x.Customer_volume / loyal_customer_volume) * 100 if x.loyal_status == 'Loyal' else (x.Customer_volume / non_loyal_customer_volume) * 100,
    axis=1
)

# Drop Customer_volume column
df = df.drop(columns=['Customer_volume'])

# Pivot table to prepare for plotting
df_pivot = df.pivot(index='loyal_status', values='Customer_percent', columns='Cost_average_bin')

# Plot the 100% stacked bar chart
df_pivot.plot(kind='bar', stacked=True, figsize=(10, 6), title = 'Stacked Bar Chart Illustrating the Distribution of Customers by Loyalty Status and Average Purchase Cost Ranges')
```

<img width="1121" height="621" alt="image" src="https://github.com/user-attachments/assets/7bc0466b-c698-491e-85e6-7b93996b8635" />

📌 **Definition**

Cost_Average_Bin: Average Order Value Segmentation. This variable represents customer segments based on their average spending per order.

Example: q1 < 12 -> This group includes customers who spend less than $12 per transaction on average

💡 **Insights & Action**

Loyal customers typically make frequent purchases, but each transaction has a relatively low average value (e.g., less than $18 per order).

This suggests they are likely individual consumers who prefer lower-priced product segments, yet demonstrate strong loyalty and long-term value.

-> Actions:

* Attract and retain this promising segment through targeted advertising and personalized marketing strategies

* Highlight affordable bundles, loyalty rewards, or exclusive low-cost offers to maintain engagement

* Position this group as a strategic base for consistent revenue and sustainable growth

## 🧮 Key Takeaways

- RFM analysis helps businesses understand customer behavior and lifetime value.

- Segmenting customers enables tailored marketing strategies, better resource allocation, and improved customer retention.

- Automation via Python ensures scalability as the company grows.


