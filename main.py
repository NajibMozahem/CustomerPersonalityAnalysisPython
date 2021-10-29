import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

the_data = pd.read_csv("data/marketing_campaign.csv", delimiter='\t')
the_data.head()
the_data.isna().sum()
# replace missing income by median values
income_median = the_data["Income"].median()
the_data = the_data.replace(np.nan, income_median)
# let us look at income in more detail
the_data["Income"].hist(bins=50)
# clearly it is skewed. It would be a good idea to take the log of this variable
the_data["Income"] = np.log(the_data["Income"])
the_data["Income"].hist(bins=50)
the_data["Income"].plot(kind="box")
# there are outliers, but I see no reasons for eliminating them
the_data.hist()
# it seems that some variables take on only a single value. Let us look at the standard deviation
the_data.describe().loc["std"]
# we see that Z_CostContact and z_Revenue have 0 standard deviation. This means that they provide
# no information since they do not vary. Let us remove them
the_data = the_data.drop(["Z_CostContact", "Z_Revenue"], axis=1)
# from the histograms produced earlier, it seemd that the variable year had a long left tail
# let us check it out
the_data["Year_Birth"].plot(kind="box")
# we see that there are outliers with some people being extremely old. They seem to be too old
# to be included. Let us remove these observations
the_data = the_data[the_data["Year_Birth"] > 1920]
# there is a date column in the data set. Let us look at how the variables are stored
the_data["Dt_Customer"].head()
# the type of the column is object. Let us convert it to a date
the_data["Dt_Customer"] = pd.to_datetime(the_data["Dt_Customer"])
# let us look at the values of the object variables
the_data.select_dtypes("object").value_counts()
# we see that there are two columns. Looking at the marital status column, we see that there are
# strange values such as yolo and alone. We should rename these to more meanigful names
the_data["Marital_Status"] = the_data["Marital_Status"].replace(["YOLO", "Alone", "Absurd"], "Single")
the_data.select_dtypes("object").value_counts()
# let us collapse this variable into a single/couple variable
the_data["Marital_Status"] = the_data["Marital_Status"].replace(["Divorced", "Widow"], "Single")
the_data["Marital_Status"] = the_data["Marital_Status"].replace(["Married", "Together"], "Couple")
# we now also do the same for education. graduated or not
the_data["Education"] = the_data["Education"].replace(["Graduation", "PhD", "Master"], "Graduate")
the_data["Education"] = the_data["Education"].replace(["Basic", "2nd Cycle", "2n Cycle"], "non-Graduate")

# let us now create some useful variables
# create a total spent variable
the_data["total_spent"] = the_data[["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]].sum(axis=1)
# create a variable that records the number of months that the customer has been a customer
the_data["months_enrolled"] = (pd.to_datetime("01-01-2015") - the_data["Dt_Customer"])/np.timedelta64(1, 'M')

# now create a variable for the total number of kids at home
the_data["children"] = the_data["Kidhome"] + the_data["Teenhome"]

# now create a variable that records the percent of purchases made online
purchases = the_data[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].sum(axis=1)
the_data["technology"] = the_data["NumWebPurchases"] / purchases
# the denominator might be zero in some cases leading to na values
the_data["technology"].isna().sum()
# we confirm that there are null values. Remove these records
the_data = the_data.dropna()

# visualization
# histogarm of total amount spent
the_data["total_spent"].hist(bins=50)
# the variable is skewed as expected
the_data["total_spent"] = np.log(the_data["total_spent"])
# look at differences in income between some groups
the_data.boxplot(column="Income", by="Education")
# who spends more, single people or couples?
the_data.boxplot(column="total_spent", by="Marital_Status")
# look at differences in spending wioth respect to number of kids
the_data.boxplot(column="total_spent", by="children")
# is there a relationship between income and amount spent?
the_data.plot(x="Income", y="total_spent", kind="scatter")
# let us produce the same graph but with a loess smoother
# remeber that we have taken the log of both of these variables
sns.regplot(x="Income", y="total_spent", data=the_data, lowess=True, ci=None)

# let us look at the variables that most correlate with amount spent
corr_matrix = the_data.corr()
corr_matrix["total_spent"].sort_values(ascending=False).drop("total_spent", axis=0).plot(kind="bar")

# cluster analysis
# first we keep the variables that we are interested in
x = the_data[["Year_Birth", "Education", "Marital_Status", "Income", "children", "Recency",
              "NumDealsPurchases", "technology", "Complain", "total_spent", "months_enrolled"]]
# we need to convert categorical variables to numbers and we need to standardize other variables

mapper = DataFrameMapper([
    (["Year_Birth"], StandardScaler()),
    (["Income"], StandardScaler()),
    (["children"], StandardScaler()),
    (["Recency"], StandardScaler()),
    (["NumDealsPurchases"], StandardScaler()),
    (["technology"], StandardScaler()),
    ("Complain", None),
    (["total_spent"], StandardScaler()),
    (["months_enrolled"], StandardScaler()),
    ("Education", LabelBinarizer()),
    ("Marital_Status", LabelBinarizer())
], df_out=True)
x_transformed = mapper.fit_transform(x)

sse = []
for k in range(1, 11):
    k_means = KMeans(n_clusters=k)
    k_means.fit(x_transformed)
    sse.append(k_means.inertia_)
plt.plot(range(1, 11), sse, '-o')
# seems that 3 or 4 is the optimal number of clusters

silhouette_coefficients = []
for k in range(2, 11):
    k_means = KMeans(n_clusters=k)
    k_means.fit(x_transformed)
    score = silhouette_score(x_transformed, k_means.labels_)
    silhouette_coefficients.append(score)
plt.plot(range(2, 11), silhouette_coefficients, '-o')

# we see that three seems to be the optimal number
k_means = KMeans(n_clusters=3)
k_means.fit(x_transformed)
yhat = k_means.predict(x_transformed)
the_data["clusters"] = yhat

# produce variables most correlated with  the clusters
corr_matrix = the_data.corr()
corr_matrix["clusters"].drop("clusters", axis=0).sort_values(ascending=False).plot(kind="bar")

sns.displot(the_data, x="Income", hue="clusters", palette="viridis")

sns.displot(the_data, x="total_spent", hue="clusters", palette="viridis")

colors = {0:'red', 1:'green', 2:'blue'}
fig, ax = plt.subplots()
for key, group in the_data.groupby("clusters"):
    group.plot(x="Income", y="total_spent", kind="scatter", ax=ax, label=key, color=colors[key], alpha=0.2)
# same can be plotted like this
# sns.scatterplot(x="Income", y="total_spent", data=the_data, hue="clusters")

the_data[["technology", "clusters"]].boxplot(column="technology", by="clusters")

clusters_marital = the_data[["clusters", "Marital_Status"]].groupby(["clusters", "Marital_Status"]).size().reset_index(name="counts")
clusters_marital = clusters_marital.pivot(index="clusters", columns="Marital_Status", values="counts")
clusters_marital.plot(kind="bar", stacked=True, rot=0)

clusters_education = the_data[["clusters", "Education"]].groupby(["clusters", "Education"]).size().reset_index(name="counts")
clusters_education = clusters_education.pivot(index="clusters", columns="Education", values="counts")
clusters_education.plot(kind="bar", stacked=True, rot=0)

clusters_children = the_data[["clusters", "children"]].groupby(["clusters", "children"]).size().reset_index(name="counts")
clusters_children = clusters_children.pivot(index="clusters", columns="children", values="counts")
clusters_children.plot(kind="bar", stacked=True, rot=0)

the_data[["NumDealsPurchases", "clusters"]].boxplot(column="NumDealsPurchases", by="clusters")

the_data[["months_enrolled", "clusters"]].boxplot(column="months_enrolled", by="clusters")

the_data[["Complain", "clusters"]].groupby(["clusters"]).sum().plot(kind="bar", rot=0)