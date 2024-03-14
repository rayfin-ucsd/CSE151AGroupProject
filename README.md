# CSE 151A Project - Group Report

**Authors**

Rehan Ali <rmali@ucsd.edu>,
Aritra Ghosh <a7ghosh@ucsd.edu>,
Scott Webster <s1webste@ucsd.edu>,
Peter Chang <p7chang@ucsd.edu>,
Yuhang Jiang <yuj052@ucsd.edu>,
Milo Nguyen <mtn022@ucsd.edu>,
Trevor Tran <trt004@ucsd.edu>,
Michael Lue <mlue@ucsd.edu>,
Jeffrey Do <jtdo@ucsd.edu>,
Joseph Kan <jokan@ucsd.edu>.

**Date**
03-13-2024

## A. Introduction

This project aims to explore the "U.S. Government Revenue Collections" dataset from Kaggle, and use it in supervised machine learning models to predict future revenues.

The data give insights to where the federal revenues come from, and how much come from tax contributions or national services, etc. A good prediction of future revenues compared to budget spending in a given year can help the government address the federal deficit. Not to mention, accurately forecasting revenue allows for more informed policy decisions, ultimately leading to efficient government spending and investment. These insights can help guide and enhance revenune collecting strategies, ensuring financial stability and growth.



We hypothesize that certain categories and channel types, as well as the date of collection affect the net revenue amount collected.

## B. Figures

(any suggestion for key figures?)

## C. Methods

### Exploratory data analysis (EDA)

- We explored the shape and types of the data, unique values, missing entries. etc.
- We also used heatmaps to visualize correlations between multiple variables.
- Frequency tables was used to show the distribution of categorical data points.
- Scatterplot was used to identify data points that deviate significantly from the majority.
- We used a line chart to investigate Net Collection Amounts by Fiscal Year.

### Data Preprocessing

- Checked for null values and dropped duplicate entries.
- Dropped columns: 'Source Line Number', 'Eletronic Category Description', 'Channel Type Description', 'Tax Category Description'.
- 'Record Date' feature discretized into an integer. For example, the oldest entry is at day 0, and the latest entry is at day 7026. This will help us model the relationship between the date of collection and the revenue (target class).
- Removed string features and kept encoded integers including the 'Fiscal Quarter Number'.

### Method 1:

### Method 2:

### Method 3:

## D. Results

### Exploratory data analysis (EDA)

- Shape of our data: `(81248, 15)`
- There is no missing entry or null values in our dataset
- Data Types:

```
Record Date                         object
Electronic Category Description     object
Channel Type Description            object
Tax Category Description            object
Net Collections Amount             float64
Electronic Category ID               int64
Channel Type ID                      int64
Tax Category ID                      int64
Source Line Number                   int64
Fiscal Year                          int64
Fiscal Quarter Number                int64
Calendar Year                        int64
Calendar Quarter Number              int64
Calendar Month Number                int64
Calendar Day Number                  int64
dtype: object
```

- Categorical Distributions

| Channel Type Description | Count |
| ------------------------ | ----- |
| Bank                     | 12443 |
| Internet                 | 13911 |
| Mail                     | 20345 |
| Other                    | 66    |
| Over-the-Counter (OTC)   | 34483 |

| Tax Category Description | Count |
| ------------------------ | ----- |
| IRS Non-Tax              | 18545 |
| IRS Tax                  | 24843 |
| Non-Tax                  | 37860 |

- Net Collections Amount distribution among categories

| Electronic Category Description | Net Collections Amount |
| ------------------------------- | ---------------------- |
| Electronic Settlement           | 5.683999e+12           |
| Fully Electronic - All          | 6.248116e+13           |
| Fully Electronic - FS           | 6.513066e+11           |
| Non-Electronic                  | 4.469666e+12           |

| Channel Type Description | Net Collections Amount |
| ------------------------ | ---------------------- |
| Bank                     | 1.574290e+13           |
| Internet                 | 4.660688e+13           |
| Mail                     | 8.189791e+12           |
| Other                    | 2.569657e+07           |
| Over-the-Counter (OTC)   | 2.746536e+12           |
|                          |                        |

| Tax Category Description | Net Collections Amount |
| ------------------------ | ---------------------- |
| IRS Non-Tax              | 2.529981e+09           |
| IRS Tax                  | 6.110596e+13           |
| Non-Tax                  | 1.217764e+13           |

- Visualisations

![Pasted image 20240313191005.png](assets/Pasted%20image%2020240313191005.png)
Figure n. Heatmap exploring correlations between data points

![Pasted image 20240313185030.png](assets/scatterplot1.png)
Figure n. Scatterplot visualizing the distribution of Net Collections Amount from 2004 - 2024

![Pasted image 20240313191108.png](assets/Pasted%20image%2020240313191108.png)
Figure n. Average Net Collections Amount by Fiscal Year

### Data Preprocessing

![image](https://github.com/rayfin-ucsd/CSE151AGroupProject/assets/44584722/66163b03-11c5-4f47-9f82-c7075260be4a)

Figure n. DataFrame after preprocessing

### Method 1:

### Method 2:

### Method 3:

- Present the key findings of your analysis.
- Use visualizations (charts, graphs) to effectively communicate the results.
- Include tables or metrics to quantify your findings when appropriate.
- Discuss the performance of your models or the validity of your hypothesis.

## E. Discussion

### Exploratory data analysis (EDA)

The frequency tables reveal that most of the revenue comes from certain categories (IRS Tax and Non-Tax) which confirms our hypothesis that certain avenues are more profitable than others.

The heatmap in Figure n finds that the Net Collections Amount is strongly correlated to the different category IDs, which means certain transaction categories bring in more revenue than others.

The scatterplot of Net Collections Amount from 2004 - 2024 reveals that the majority of collections made were small, with a sparse number of outlying large collections. There is a noticeable trend that these outliers increase in count and magnitude each passing year.

This poses the questions: does this correspond to the increasing income inequality? Or did the government become more effective at collecting taxes?

To understand the trend further, we plotted a line chart showing the average Net Collections Amount by fiscal year. This revealed a pattern consistent with a post-pandemic economic recovery. However, a closer look suggests a more nuanced story.

The significant rise in collections could be partially attributed to the growth of specific sectors like e-commerce (e.g., Amazon) and video conferencing (e.g., Zoom) that boomed during COVID-19. The dip at the end of 2021, reflects the decreasing revenue of the above sectors, which might be the reason for multiple layoffs since then.

It's important to acknowledge limitations here. A single data point (fiscal year) might be influencing the observed linear trend. Additionally, Net Collections encompass various sources beyond just income tax.

### Data Preprocessing

We hypothesized that the Fiscal Quarter would be a helpful feature in predicting the revenue, so we kept the information.

We also kept the columns: Calendar Year Number, Calendar Quarter Number, Calendar Month Number, Calendar Day Number. But these features are not independent of each other, so we have several options for our analysis of their relationships:

1. Record Date Discretized
2. Calendar Year Number, Calendar Month Number, Calendar Day Number
3. Calendar Year Number, Calendar Quarter Number,
4. Others

### Method 1:

### Method 2:

### Method 3:

## F. Conclusion

- Summarize the main takeaways of your project.
- This is where you do a mind dump on your opinions and possible future directions. Basically what you wish you could have done differently. Here you close with final thoughts

## G. Collaboration

- Rehan Ali - Created Discord Server, helped with organization, performed a variety of analysies in the EDA section and helped write significant parts of the readme during all the milestones. 
- Aritra Ghosh - Title:
- Scott Webster - Title:
- Peter Chang - Title:
- Yuhang Jiang - Title:
- Milo Nguyen - Title:
- Trevor Tran - Title:
- Michael Lue - Title:
- Jeffrey Do - Title:
- Joseph Kan - Title:
