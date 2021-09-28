# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 15:04:31 2021

@author: Rony's PC
"""



"""
1. Introduction
---------------

This is my second kernel at Kaggle. I choosed economic data tabular which 
is a good way to introduce feature engineering and ensemble modeling. 
Firstly, I will display some feature analyses than i will focus on the 
feature engineering. Last part concerns modeling and predicting sales 
marketing as using an voting procedure. As an economic label problem, 
I prefer to do a regression model Random forest Grid Search with 
Cross Validation.

This script follows three main parts:

    Feature analysis
    Feature engineering
    Modeling
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
import seaborn as sns





"""
2. Load and check data
----------------------
2.1 Load data
"""

x_features = pd.read_excel(r"C:/Users/Rony's PC/OneDrive/ML/miniproject1/Features data set.xlsx") # features 
y_sales = pd.read_csv(r"C:/Users/Rony's PC/OneDrive/ML/miniproject1/sales data-set.csv.zip") # labels
z_stores = pd.read_csv(r"C:/Users/Rony's PC/OneDrive\ML/miniproject1/stores data-set.csv") # stores

"""
2.2 Inquering all Data
"""

print(x_features.head())
print(y_sales.head())
print(z_stores.head())
print('-'*10)
print('\n')

"""
After looking the data we have a problem with missing data
So to fix it we need apply the dates in both columns to same type date.
I choose as best as I can the featurs that fit the economic aspects predict.
Following this vision: 
    1)I didn't take data size stores
    2) I merge only the full cells data
"""


"""
2.3 Groupby working
The groupby drop all missing rows
""" 

y_sales['Date'] = pd.to_datetime(y_sales['Date'])
# optimize the sales via store in a unique date with all Deparments to be preper 
y_sales = y_sales.groupby(['Store', 'Date']).sum()
# we have to repeat its again to fit x_features y_sales 
x_features['Date'] = pd.to_datetime(x_features['Date'])
x_features = x_features.groupby(['Store', 'Date']).sum()
print(x_features.dtypes)
print(y_sales.dtypes)
print(z_stores.dtypes)
print('-'*10)
print('\n')

"""
2.4 Merging all Data
As an inner mean - marge just the common Datas.
"""

Combined_table = pd.merge(x_features, y_sales['Weekly_Sales'], how='inner', right_index=True, left_index=True)
Combined_table.isna().sum()
Combined_table.info
Combined_table['Weekly_Sales'].describe()
Combind_graf = Combined_table.copy()

# Very well... It seems that your minimum price is larger than zero. Excellent!

"""
2.5 inquery by: Skewness @ skew
"""
print("Skewness: %f" % Combined_table['Weekly_Sales'].skew())
print("Kurtosis: %f" % Combined_table['Weekly_Sales'].kurt())
"""
2.5.1) Have appreciable positive skewness.
2.5.2) Have appreciable positive skewness.
"""



"""
3 Inquery by Plotting 
---------------------
3.1 Histogram plot
"""
# let see how Weekly_Sales corellative to other features ?
# sns.distplot(Combined_table['Weekly_Sales'])
sns.displot(data = Combined_table, x = 'Weekly_Sales', kde=True)

"""
3.2 Inquery of outlier by box plot store / Weekly_Sales
"""
# first convert the store from index into a column
# Takea a look over each sotre outlayers via "Weekly_Sales'.
Combined_table = Combined_table.reset_index(level=0)
var = 'Store'
data = pd.concat([Combined_table['Weekly_Sales'], Combined_table[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="Weekly_Sales", data=data)
fig.axis(ymin = 180000, ymax = 4000000)

# By inquering with "Boxplot" I see that all thefeatures are outlier.

"""
3.3 scatterplot - Data Visualization
"""
# Let display the relationship between two numerical variables
# For any combination features. 
sns.set()
cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Weekly_Sales']
sns_plot = sns.pairplot(Combind_graf[cols].sample(100), height = 2.5) 
plt.show()

# Here by 'Scatterplot' I could see the relations between all features. The calculating takes a lot of time so it's suggested to take small number sample. Finally, we haven't correlation between the features

"""
3.4 Plotting all features via the label (prediction of sales)
"""
# I try to see if it's any corralation between the features
Combind_graf[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Weekly_Sales']].plot(subplots=True, figsize=(20,15))
plt.show()

# In other way plotting all features again with 'Combind_graf'  no relations between all of them.



"""
4. Feature analysis
-------------------

4.1 Plotting Confusional Correlation matrix between numerical values.
"""
# In the following Matrix we will see how mach the features are confusing?.
# As we see not all features are not corelative. 
g = sns.heatmap(Combind_graf[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Weekly_Sales']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
plt.show()

"""
4.2 Plotting Confusional Correlation matrix with no features correlative.
"""
# Since I have two Prameters with correlasition, I drop one of them - 'MarkDown1',
# Why ?
# To prevent  Linkage featurs that are corelatived.
g = sns.heatmap(Combind_graf[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Weekly_Sales']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
plt.show()

"""
conclusion: Finally The linckage is no so big.
----------
After checking the accuracy with and without 'MarkDown1' the different
 seem to be small
"""
Combined_table = Combined_table.drop(['MarkDown1'], axis = 1)


"""
5. Feature engineering
----------------------
"""
"""
5.1 Since I must take some economic vision for my analyzation. I prefer to 
take only two tables for my inquiry. The two table are: 1) 'features', 2) 
'sales' that correspond to our label. It was done with economic aspect and 
it's the reason why I don't take 'stores'. I combine the 'features' and 'sales' 
and split them to be : features and label
"""
"""
5.2 Just to be on the safe side. As we could see, I take some types plots for 
features inquiry. Some plots show no relations and one other show that those 
two features are in similarity. If it does maybe I have a linkage that could 
influence my accuracy. So, I check the output accuracy with and without this 
feature - no different. So, I prefer even that to reduce it.
"""
"""
5.3 The data are not so big, so maybe the training could be problematic. 
As a model of economic predicts, I can't do any augmentation with no meaning. 
Other problem corresponds to lost time once the combine tables were done with 
no empty rows and columns. I prefer the bagging to be 3 (cv = 3) - more the 
bagging greatest more time calculating is bigger but maybe more accreditable. 
Each individual tree in the random forest spits out a class prediction and the 
class with the most votes become our model’s prediction (see figure below).
"""


"""
6.Modeling
----------
"""

# Definition of Y (label) and X (inputs)
# Define label for the new merge table: "combined_table"
# The 'Weekly_Sales' is Indexial so it dosn't take as a label
y = Combined_table['Weekly_Sales']
# Define features for the new merge table: "combined_table"
x = Combined_table.drop(['Weekly_Sales'], axis=1)


"""
6.1 preprocessing normalization values between 0 and 1
"""
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
y.shape
y = y.values.reshape(4320, 1)
y_scaled = scaler.fit_transform(y)
x.head()
x.tail()

"""
6.2 splitting the Datas to train @ test 
"""
# we need to take x_scaled after being notmalized
# In the first step we will split the data in training and remaining dataset
x_train, x_valid, y_train, y_valid = train_test_split(x_scaled, y, train_size=0.80, random_state=42)

"""
6.3 splitting the test to test @ val
"""
# Now since we want the valid and test size to be equal (10% each of overall data). 
# we have to define valid_size=0.5 (that is 50% of remaining data)
test_size = 0.5
x_test, x_valid, y_test, y_valid = train_test_split(x_scaled,y, test_size=0.5)


# # we need to add an additional dimantion to get numpy arrey to keras
# x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1]) 
# x_valid = x_valid.reshape(x_valid.shape[0],1,x_valid.shape[1])
# x_test = x_test.reshape(x_test.shape[0],1,x_test.shape[1]) 

Test_Data = (x_test, y_test)

print('-'*10)
print('\n')
print(x_train.shape), print(y_train.shape)
print(x_valid.shape), print(y_valid.shape)
print(x_test.shape), print(y_test.shape)
print('-'*10)
print('\n')

"""
7. Random forest and Adaboost
-----------------------------
"""


"""
7.1 Random Forest
Are an ensemble learning method for classification, regression. In a Random Forest, each tree has an equal vote on the
final decision. In a Random Forest each decision tree is made independently of the others.
For regression tasks, the mean or average prediction of the individual trees is returned.
Random forests generally outperform decision trees, but their accuracy is lower than gradient boosted trees.
However, data characteristics can affect their performance.
"""
random_state = 2
classifiers = []


random_forest = RandomForestRegressor() 
random_forest.fit(x_train, y_train.ravel())
y_pred = random_forest.predict(x_test) 

random_forest_tuning = RandomForestRegressor(random_state = random_state)
param_grid = {'n_estimators': [100, 200, 500],'max_features': ['auto', 'sqrt', 'log2'],'max_depth' : [4,5,6,7,8],'criterion' :['mse', 'mae']}


GSCV = GridSearchCV(estimator=random_forest_tuning, param_grid=param_grid, cv = 3)
# By grid serch let look wich are the best choice to be taken ?.
GSCV.fit(x_train, y_train.ravel())
GSCV.best_params_ 

random_forest = RandomForestRegressor(random_state = random_state)
random_forest.fit(x_train, y_train.ravel())
y_pred_random = random_forest.predict(x_test)
print('MAE: ', mean_absolute_error(y_test, y_pred))
print('MSE: ', mean_squared_error(y_test, y_pred))


random_forest_out_of_bag = RandomForestRegressor(oob_score=True)
random_forest_out_of_bag.fit(x_train, y_train.ravel())
print('Random forest score')
print(random_forest_out_of_bag.oob_score_) 

"""
Conclusion
I cannot see any feature to be predictable for sales but all the features
together can be predictable that no growing is expectable. Else Random Forest
show accuracy of around 94%.
"""






















