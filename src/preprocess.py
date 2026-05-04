import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from features import CombinedAttributesAdder

def get_preprocessor(num_cols, cat_cols):
    num_pipeline = Pipeline([
        ('Imputer', SimpleImputer(strategy='median')), 
        ('AttribsCreator', CombinedAttributesAdder()),
        ('Scaler', StandardScaler())
        ])
    cat_pipeline = Pipeline([
        ('Imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('OneHotEncoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_cols), 
        ('cat', cat_pipeline, cat_cols)
    ])
    return full_pipeline

