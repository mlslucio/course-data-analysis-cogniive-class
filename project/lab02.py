import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()

# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)
#plt.show()

#correlation between price and engine-size
print(df[["engine-size", "price"]].corr())

#highway-mpg as potential predictor of price
sns.regplot(x="highway-mpg", y="price", data=df)
#plt.show()

print(df[['highway-mpg', 'price']].corr())

#weak linear relantionship
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
#plt.show()

#boxplot
sns.boxplot(x="engine-location", y="price", data=df)
#plt.show()

sns.boxplot(x="drive-wheels", y="price", data=df)
#plt.show()

print(df.describe())


drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts

print(drive_wheels_counts)

#grouping
df_group_one = df[['drive-wheels','body-style','price']]

df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
print(df_group_one)

#pearson correlation
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#ANova
grouped_test2=df[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.head(2)

f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'],
                              grouped_test2.get_group('4wd')['price'])

print("ANOVA results: F=", f_val, ", P =", p_val)

