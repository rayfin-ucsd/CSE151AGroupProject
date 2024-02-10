# CSE151AGroupProject
Group Project for 151a

Last updated 2/10/2024 at 2:29 PM

Group Members: 

Rehan Ali <rmali@ucsd.edu>
Aritra Ghosh <a7ghosh@ucsd.edu>,
Scott Webster <s1webste@ucsd.edu>,
Peter Chang <p7chang@ucsd.edu>,
Yuhang Jiang <yuj052@ucsd.edu>,
Milo Nguyen <mtn022@ucsd.edu>,
Trevor Tran <trt004@ucsd.edu>,
Michael Lue <mlue@ucsd.edu>,
Jeffrey Do <jtdo@ucsd.edu>,
Joseph Kan <jokan@ucsd.edu>


***** INTRODUCTION *****

We ended up picking the "U.S. Government Revenue Collections" dataset from
Kaggle. Our model's target remains the same: predicting future revenue 
collections based on the model's features.


***** PREPROCESSING *****

Our data was relatively clean, so there wasn't too much to do here. 
We checked for null values (of which there were none), dropped duplicates, 
and got rid of columns that were redundant. The electronic category 
description, channel type description, and tax category description were 
already label encoded in separate columns, so those could be dropped. 
Record date was also split into day, month, and year in other columns, so we 
dropped that too. We discretized our record dates as datetime objects by # of days.
(For example, the oldest entry is at day 0, and the latest entry is at day 7026)
We made sure that each duplicate column matched one another before deleting them. 

We first wanted to see how fiscal quarters were related to eachother by year. 
Finding a "not large" standard deviation between quarters, we decided that this 
fiscal data was evenly distributed, so we could useit without further preprocessing. 


***** DATA EXPLORATION *****

We became "one with the data." Shapes, data types, unique values per feature, unique days,
entries per year, net collections amount by year, net collections amount by electronic 
category, channel type, and tax category, etc. etc. etc. 

We then shoved all of these into a heatmap to look for correlations between features, finding 
that net collections amount is correlated pretty strongly with the different category IDs, 
which tells us that certain categories of transaction give more revenue that others. A couple
 bar charts reveal that a majority of revenue is associated with certain categorical descriptions. 
 This makes sense intuitively: certain avenues are more profitable that others. 
 Correlating these descriptions could help us make predictions!
 
 We then made a scatterplot of all net collections amount by date. This only tells us that a majority
 of collections made were small, with a sparse number of outlying large collections. There is a noticeable 
 trend that these outliers increase in count and magnitude as years go on. 
 
 Is this commensurate with growing income inequality? Or maybe the government got better at collecting?
 
 Investigating further, we made a line-plot of average net collections amount by fiscal year. This showed a linear increase 
 with a noticeable drop around 2021. Thinking about this intuitively requires us to consider the American economy and political 
 atmosphere, which we shall do sparsely: the impact of COVID-19 caused a temporary recession followed by massive growth in the upper 
 quartiles of wealth and industry. (Amazon and Zoom, for example, thrived in this period.) This was not without a cost: income 
 inequality grew drastically, so while the economy grew, people got *poorer*- meaning the large number of concentrated small collections 
 decreased in magnitude.
 
 Or, it could just be some outlier bending our curve. It might be tempting to interpret these graphs based on income, but these collections 
 cover a lot more than just income tax. Our future work in prediction modeling may reveal more about the root cause of this trend.
(We could also do extra work that involves more datasets, but that's beyond the scope of this project.) 
 

 
