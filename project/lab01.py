import pandas as pd
import numpy as np

filename = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(filename, names = headers)

#convert '?' to NaN
df.replace("?", np.nan, inplace = True)

#verifyng missing data
missing_data = df.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")

#replace null columns

#replace normalized-losses column with avg
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)

df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

#replace bore column with avg
avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)

df["bore"].replace(np.nan, avg_bore, inplace=True)

#replace horsepower column with avg
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)

df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

#replace peak-rpm column with avg
avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)

df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)

#replace number os doors null values based on frequency
numDoors = df['num-of-doors'].value_counts().idxmax()

df["num-of-doors"].replace(np.nan, numDoors, inplace=True)

#drop null price columns, because price is the target column
df.dropna(subset=["price"], axis=0, inplace=True)

df.reset_index(drop=True, inplace=True)

#format data with correct values
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()

#binning horsepower data
df["horsepower"]=df["horsepower"].astype(int, copy=True)

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)

group_names = ['Low', 'Medium', 'High']

df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )

print(df[['horsepower','horsepower-binned']].head(20))

print(df["horsepower-binned"].value_counts())

#set dummy variables for fuel-type
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
df = pd.concat([df, dummy_variable_1], axis=1)

df.to_csv('clean_df.csv')








