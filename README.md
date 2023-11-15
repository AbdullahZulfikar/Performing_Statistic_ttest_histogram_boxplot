# Performing_Statistic_ttest_histogram_boxplot
in this, i have used Scipy to deal with the states data matplot,plotly and seaborn

libraries:
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

importing the pandas and making  the line graph between the data fo two clinic
df = pd.read_csv('data/annual_deaths_by_clinic.csv')
fig = px.line(df,
              x='year',
              y='deaths',
              color='clinic',
              title="total year birth by clinic")

fig.show()

calculating the average and adding them to a new column
df[ "pct_deaths"] = df['births']/df["births"].sum() * 100
df["maternal_death_rate"] = (df['deaths']/df['births']) * 100

plotting to line plot
fig = px.line(df,
              x='year',
              y='pct_deaths',
              color='clinic',
              title="total year birth by clinic")

fig.show()

this part of the code is to make the two data points points before the washing ands and after the washing hand and calculating the averages

df_m['pct_deaths'] = df_m.deaths/df_m.births

before_washing = df_m[df_m.date < handwashing_start]
after_washing = df_m[df_m.date >= handwashing_start]

bw_rate = before_washing.deaths.sum() / before_washing.births.sum() * 100
aw_rate = after_washing.deaths.sum() / after_washing.births.sum() * 100
before_washing['date'] = pd.to_datetime(before_washing['date'])
after_washing['date'] = pd.to_datetime(after_washing['date'])


calculating the rolling mean 

roll_df = before_washing.set_index('date')
roll_df = roll_df.rolling(window=6).mean()
df_m["date"] = pd.to_datetime(df_m["date"])


the main part and the messy part 
in the xaxis locator to set major I have used the madates function form matplot.dates
this will get the year data from the dataframe and plot on the ticker as major
same goes for formate and minor locator
setting limit by getting the min date and max date in the data so our range would be between these limits
to use the legend you must give the labels to the plot or any graph you have made above 

plt.figure(figsize=(14,8), dpi=200)
plt.title('Percentage of Monthly Deaths over Time', fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14, rotation=45)
plt.ylabel('Percentage of Deaths', color='crimson', fontsize=18)
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.set_xlim([df_m.date.min(),df_m.date.max()])
plt.grid(color="grey",linestyle="--")

ma_line = plt.plot(roll_df.index, 
                    roll_df.pct_deaths, 
                    color='crimson', 
                    linewidth=3, 
                    linestyle='--',
                    label='6m Moving Average'),
bw_line = plt.plot(before_washing.date, 
                    before_washing.pct_deaths,
                    color='black', 
                    linewidth=1, 
                    linestyle='--', 
                    label='Before Handwashing'),
aw_line = plt.plot(after_washing.date, 
                    after_washing.pct_deaths, 
                    color='skyblue', 
                    linewidth=3, 
                    marker='o',
                    label='After Handwashing')

plt.legend(fontsize=18)

plt.show()


this is to make the boxplot we use in Statistic washing hand is the new colum with the no condition in this case
import numpy as np
df_m["washing_hand"] = np.where(df_m.deaths < int(handwashing_start), "no", "yes")


# Concatenate the two columns to use as x-axis in the box plot


box = px.box(df_m,
       x="washing_hand",
       y="pct_deaths",
       color="washing_hand",
       title="How did state change after washing ands")
box.update_layout(xaxis_title="Washing hands",
           yaxis_title="Percentage of Monthly Deaths")
box.show()


plotting histogram
#now making the histogram
color = ['blue',"red",'purple',"black"]
fig = px.histogram(df_m,
                   x="date",
                   y="pct_deaths",
                   title="Number of deaths per year",
                   color_discrete_sequence=color,
                   opacity=0.3,
                   nbins=30,
                   barmode="overlay",
                   histnorm="percent",
                   marginal='box')
fig.update_layout(bargap=0.01)
fig.show()

using the visualization form the seaborn.kdeplot between after and before washing 
plt.figure(figsize=(14,8))
sns.kdeplot(before_washing.pct_deaths,
            shade=True,
            clip=(0,1),
            legend=True)
sns.kdeplot(after_washing.pct_deaths,
            shade=True,
            clip=(0,1),
            legend=True)
plt.title('Est. Distribution of Monthly Death Rate Before and After Handwashing')
plt.xlim(0, 0.40)
plt.show()

and finally perform the ttest by using the Scipy 
from scipy import stats

t_test , p_value = stats.ttest_ind(a=before_washing.pct_deaths,
                                   b=after_washing.pct_deaths)
print(t_test)
print(p_value)
