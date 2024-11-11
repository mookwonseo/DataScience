## This is my summary for the data analysis/science
## Based on Scikit Learn, pandas, and numpy.
## I tried to combine sources from Kaggle, Coursera, and Edx

import os
import pandas as pd
## Model selection. 
## train_test_split: split the data set into train and test sets
from sklearn.model_selection import train_test_split
## cross_val_score: 
# split the data set into a certain number of train sets (different)
# and a test set to do the cross validation 
# Benefits
# Preventing the overfiitng and underfitting
# Hyperparameter tuning
# Robustness to outliers. 
from sklearn.model_selection import cross_val_score

## Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

## Regression Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

## Metric
from sklearn.metrics import mean_absolute_error


## Check the file path.
if not os.path.exists("./input/train.csv"):
    ValueError("File does not exist.")

## Return the prediction score. 
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators = 100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

## Read a file. 
X_full = pd.read_csv('./data/train.csv',index_col='Id')

## Drop NaN values on the y value. 
X_full.dropna(subset=['SalePrice'],axis=0,inplace=True)
## Set y.
y = X_full.SalePrice
## Exclude y column from X
X_full.drop(['SalePrice'],axis=1,inplace=True)

## Split the test set into train and validate sets. 
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)

## Split the categorical columns and numerical columns
categorical_cols = [col for col in X_train_full.columns if X_train_full[col].dtype == 'object' and X_train_full[col].nunique() < 10]
numerical_cols = [col for col in X_train_full.columns if X_train_full[col].dtype in ['int64','float64']]
## Other ways to figure out the categorical cols and numerical cols
# temp_categorical = X_train_full.select_dtypes('object')
# temp_categorical_cols = temp_categorical.loc[:,temp_categorical.nunique()<10].columns
# diff1 = set(temp_categorical_cols) - set(categorical_cols)
# temp_numerical_cols = X_train_full.select_dtypes(exclude='object').columns
# diff2 = set(temp_numerical_cols) - set(numerical_cols)
# print('Different two columns: ', diff1, diff2)

