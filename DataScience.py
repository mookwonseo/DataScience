## This is my summary for the data analysis/science
## Based on Scikit Learn, pandas, and numpy.
## I tried to combine sources from Kaggle, Coursera, and Edx

import os
import pandas as pd
import numpy as np
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

# Combine two columns together
my_cols = categorical_cols + numerical_cols

# Preprocessing for numerical data using Scikit Learn

cat_X_train = X_train_full[categorical_cols]
cat_X_valid = X_valid_full[categorical_cols]
num_X_train = X_train_full[numerical_cols]
num_X_valid = X_valid_full[numerical_cols]

missing_val_counts = num_X_train.isnull().sum()
missing_val_columns = num_X_train.columns[missing_val_counts>0]

# Basic idea of missing values from numerical data
missingValCntrlOption = 3
strategy="constant"
try: 
    if missingValCntrlOption == 1:
        my_X_train_num = num_X_train.drop(missing_val_columns, axis = 1) # axis = 0: index, axis = 1: columns
        my_X_valid_num = num_X_valid.drop(missing_val_columns, axis = 1)
        out_string = "Drop Missing Values"

    elif missingValCntrlOption == 2:
        myimputer = SimpleImputer(strategy=strategy) #default: mean
        my_X_train_num = pd.DataFrame(myimputer.fit_transform(num_X_train))
        my_X_valid_num = pd.DataFrame(myimputer.transform(num_X_valid))
        
        my_X_train_num.columns = num_X_train.columns
        my_X_valid_num.columns = num_X_valid.columns
        my_X_train_num.index = num_X_train.index
        my_X_valid_num.index = num_X_valid.index
        out_string = "Imputed Missing Values"

        ## Other ways to relace the nan to the mean values
        ## fit_transform to compute the mean values of each column in num_X_train
        ## insert this values into num_X_train and num_X_valid.
        # temp_num_X_train = num_X_train.copy()
        # temp_num_X_valid = num_X_valid.copy()
        # mean_train =num_X_train[missing_val_columns].mean()
        # temp_num_X_train[missing_val_columns] = num_X_train[missing_val_columns].replace(np.nan, mean_train)
        # temp_num_X_valid[missing_val_columns] = num_X_valid[missing_val_columns].replace(np.nan, mean_train)
        # print(temp_num_X_train.compare(my_X_train))
        # print(temp_num_X_valid.compare(my_X_valid))
    
    elif missingValCntrlOption == 3:
        X_train_plus = num_X_train.copy()
        X_valid_plus = num_X_valid.copy()
        
        for col in missing_val_columns:
            X_train_plus[col+"_missing_vals"]=X_train_plus[col].isnull()
            X_valid_plus[col+"_missing_vals"]=X_valid_plus[col].isnull()
        
        myimputer = SimpleImputer(strategy=strategy) #default: mean
        my_X_train_num = pd.DataFrame(myimputer.fit_transform(X_train_plus))
        my_X_valid_num = pd.DataFrame(myimputer.transform(X_valid_plus))
        
        my_X_train_num.columns = X_train_plus.columns
        my_X_valid_num.columns = X_valid_plus.columns
        
        my_X_train_num.index = X_train_plus.index
        my_X_valid_num.index = X_valid_plus.index

        out_string = "Extended Imputed Missing Values"
    else:
        raise KeyboardInterrupt("Out of range")
finally:
    pass

# Basic idea of missing values from categorical data
# Check if some categorical names in test data is not in test data
# I didn't import test data set yet. So I am testing with valid set. 
good_label_cols = [col for col in categorical_cols if set(X_valid_full[col]).issubset(X_train_full[col])]
bad_label_cols = set(X_train_full[categorical_cols])-set(X_valid_full[good_label_cols])

label_X_train = cat_X_train.drop(bad_label_cols, axis = 1)
label_X_valid = cat_X_valid.drop(bad_label_cols, axis = 1)

print(X_train_full[categorical_cols])

objColumnCntrlOption = 2
try:
    if objColumnCntrlOption == 1:
        ### drop the whole columns
        my_X_train_cat = pd.DataFrame()
        my_X_valid_cat = pd.DataFrame()
    elif objColumnCntrlOption == 2:
        ## Need to fix it
        ordinal_encoder = OrdinalEncoder() # Your code here
        label_X_train[good_label_cols] = ordinal_encoder.fit_transform(label_X_train[good_label_cols])
        label_X_valid[good_label_cols] = ordinal_encoder.transform(label_X_valid[good_label_cols])

        my_X_train_cat = label_X_train
        my_X_valid_cat = label_X_train
    elif objColumnCntrlOption == 3:
        low_cardinality_cols = [col for col in good_label_cols if label_X_train[col].nunique() < 10]
        high_cardinality_cols = list(set(good_label_cols)-set(low_cardinality_cols))

        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        my_X_train_cat = pd.DataFrame(OH_encoder.fit_transform(label_X_train[low_cardinality_cols]))
        my_X_valid_cat = pd.DataFrame(OH_encoder.transform(label_X_valid[low_cardinality_cols]))

        my_X_train_cat.index = label_X_train.index
        my_X_valid_cat.index = label_X_valid.index

        my_X_train_cat.columns = label_X_train.columns
        my_X_valid_cat.columns = label_X_valid.columns
    elif objColumnCntrlOption == 4:
        ### Simple imputer most frequent
        print("Not Implemented")
    else:
        raise KeyboardInterrupt("Out of range")
finally:
    pass


#print(my_X_train_cat)



filtered_X_train = pd.concat([my_X_train_num, my_X_train_cat ], axis = 1)
filtered_X_valid = pd.concat([my_X_valid_num, my_X_valid_cat ], axis = 1)

print(score_dataset(filtered_X_train, filtered_X_valid, y_train, y_valid))

############################################################
############### Using PipeLine #############################
############################################################
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# Preprocessing for numerical data
numerical_trainsformer = SimpleImputer(strategy="constant")
# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num',numerical_trainsformer,numerical_cols),
        ('cat',categorical_transformer, categorical_cols)
    ]
)
# Define model
model=RandomForestRegressor(n_estimators=100, random_state=0)
# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model',model)
])

# Preprocessing of training data, fit model 
clf.fit(X_train, y_train)
preds = clf.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))