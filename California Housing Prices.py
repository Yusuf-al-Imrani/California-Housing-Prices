#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import tarfile
from six.moves import urllib

download_root = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = download_root + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[2]:


fetch_housing_data()


# In[3]:


import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[4]:


housing = load_housing_data()


# In[5]:


housing.head()


# In[6]:


#show some statistical insights on the data
housing.describe()


# In[7]:


#visualizing all the data by ploting all the numerical values
import matplotlib.pyplot as plt
housing.hist(bins = 50, figsize=(20,15))
plt.show()


# In[8]:


#Splitting the data
import numpy as np
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indces = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indces]

train_set, test_set = split_train_test(housing, 0.2)
len(train_set)


# In[9]:


#Ready train test split funcion
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)


# In[10]:


#Getting train, test stratified splitting

#First creating a new attribute to split based upon it(median income in our case)
housing["income_cat"] = pd.cut(housing["median_income"], 
                              bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
                              labels = [1, 2, 3, 4, 5])
housing["income_cat"].hist()

#data are ready to do stratified sampling bases on the income category
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#To chech sampling percentage of the data, let's plot the data based on appearance percentage of "median_income" attribute 
strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# In[11]:


#Removing the newly added attribute "income_cat"
for set_ in (strat_train_set, strat_test_set):
    set_.drop(["income_cat"], axis=1, inplace=True)


# In[12]:


#Visualizing the data

#first takign a copy of the data to work on visualizing on it
housing = strat_train_set.copy()

housing.plot(kind = "scatter", x = "longitude", y = "latitude",
                  alpha = 0.1, s = housing["population"]/100,
                  label = "Population", figsize = (10,7), 
                 c = "median_house_value", cmap = plt.get_cmap("jet"),
                  colorbar = True)
plt.show()


# In[13]:


#Looking for correlations
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending = False)


# In[14]:


from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize = (12, 8))


# In[15]:


#zooming more on the most correlated attribute(median_income) to our label(median_house_value)
housing.plot(kind = "scatter", x = "median_income", y = "median_house_value", alpha = 0.1)


# In[16]:


housing= strat_train_set.drop("median_house_value", axis = 1)
housing_labels = strat_train_set["median_house_value"].copy()


# In[17]:



#Function to add attributes 
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

#Creating pipline for handling  numerical attributes (missing data, add attributes, and scalling)     
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy="median")),
                    #('attribs_adder', CombinedAttributesAdder()),
                   ('attribs_adder',CombinedAttributesAdder(add_bedrooms_per_room=True)),
                    ('std_scaler', StandardScaler()),
                    ])

#Pipline for handling numerical and categorical attributes all together
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

housing_num = housing.drop("ocean_proximity", axis=1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
                                ("num", num_pipeline, num_attribs),
                                ("cat", OneHotEncoder(), cat_attribs),
                                ])

housing_prepared = full_pipeline.fit_transform(housing)


# In[18]:


housing_df = pd.DataFrame(housing_prepared)
housing_df


# In[19]:


#Selecting and training models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

LinReg = LinearRegression()
LinReg.fit(housing_prepared, housing_labels)

DT_reg = DecisionTreeRegressor()
DT_reg.fit(housing_prepared, housing_labels)

#testing predictions on first rows of the data 
instances = housing.iloc[:5]
instances_labels = housing_labels.iloc[:5]
instances_prepared = full_pipeline.transform(instances)
print("Predictions= ", LinReg.predict(instances_prepared))
print("Labels: ", list(instances_labels))


# In[20]:


#Measuring RMSE
from sklearn.metrics import mean_squared_error

#rmse for Linear Regression
LinReg_housing_predictions = LinReg.predict(housing_prepared)
LinReg_mse = mean_squared_error(housing_labels, LinReg_housing_predictions)
LinReg_rmse = np.sqrt(LinReg_mse)
print("RMSE for Linear Regression: ",LinReg_rmse)

DT_housing_predictions = DT_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, DT_housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
print("RMSE for Decisin Tree: ",lin_rmse)


# In[21]:


#Evaluating the models using Cross-Validation

#displaying scores
def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())


##Calculating scores for Decision Tree model 
from sklearn.model_selection import cross_val_score

DT_scores = cross_val_score(DT_reg, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=10)
DT_reg_scores = np.sqrt(-DT_scores)
display_scores(DT_reg_scores)


# In[22]:


#It seems not a very good model as rmse is high


# In[23]:


#Calculating scores for Linear Regression model 
LinReg_scores = cross_val_score(LinReg, housing_prepared, housing_labels, 
                           scoring= "neg_mean_squared_error", cv=10)
LinReg_rmse_scores = np.sqrt(-LinReg_scores)
display_scores(LinReg_rmse_scores)


# In[24]:


#Calculating scores for Random Forest model
from sklearn.ensemble import RandomForestRegressor

RF_reg = RandomForestRegressor()

RF_scores = cross_val_score(RF_reg, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=10)
RF_reg_scores = np.sqrt(-RF_scores)
display_scores(RF_reg_scores)


# In[25]:


#Fine tuning the model

from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

RF_reg = RandomForestRegressor()
grid_search = GridSearchCV(RF_reg, param_grid, cv=5,
                           scoring="neg_mean_squared_error",
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)


# In[26]:


grid_search.best_params_


# In[27]:


#evaluating the final model on the test set
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse


# In[ ]:




