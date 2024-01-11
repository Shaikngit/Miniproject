from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self, variables: str):

        if not isinstance(variables, str):
            raise ValueError("variables should be a list")

        self.variables = variables
        
         
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        self.modeVal = X[self.variables].mode()
        print("WeekdayImputer.fit")
        return self 

    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        X[self.variables] = X[self.variables].fillna(self.modeVal[0])
        print("WeekdayImputer.transform")           
        return X

class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, variables: str):

        if not isinstance(variables, str):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        self.fill_value=X[self.variables].mode()[0]
        print(f"WeathersitImputer.fit")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables]=X[self.variables].fillna( self.fill_value)
        print("WeathersitImputer.transform")
        return X


class Mapper(BaseEstimator, TransformerMixin):
    """Categorical variable mapper."""

    def __init__(self, variables: str, mappings: dict):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        print("Mapper.fit")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        #for feature in self.variables:
        #Code used to debug the na values in mapper , turned out it was first mapper yr causing the problem
        #hardcoded to yr values , need to check later what is the problem
        # print(X.isna().sum())
        # numeric_columns = X.select_dtypes(include=[np.number]).columns
        # print((X[numeric_columns].isna() | np.isinf(X[numeric_columns])).sum())
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)
        print("Mapper.transform")
        return X

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variables: str):
        # YOUR CODE HERE
        if not isinstance(variables, str):
            raise ValueError('variables should be a str')

        self.variables = variables
            
    def fit(self, X: pd.DataFrame, y: pd.Series =None):
        q1 = X.describe()[self.variables].loc['25%']
        q3 = X.describe()[self.variables].loc['75%']
        iqr = q3 - q1
        self.lower_bound = q1 - (1.5 * iqr)
        self.upper_bound = q3 + (1.5 * iqr)
        # we need this step to fit the sklearn pipeline
        print(f"OutlierHandler.fit")
        return self
                  
    def transform(self,X: pd.DataFrame):
        for i in X.index:
            if X.loc[i,self.variables] > self.upper_bound:
                X.loc[i,self.variables]= self.upper_bound
            if X.loc[i,self.variables] < self.lower_bound:
                X.loc[i,self.variables]= self.lower_bound
        print("OutlierHandler.transform")
        return X


    

class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, variable: str):
        # YOUR CODE HERE
        if not isinstance(variable, str):
            raise ValueError("variables should be a str")

        self.variable = variable
        self.encoder = OneHotEncoder(sparse_output=False)
        
    def fit(self,X: pd.DataFrame,y: pd.Series = None):
        # YOUR CODE HERE
        self.encoder.fit(X[[self.variable]])
        self.encoded_features_names = self.encoder.get_feature_names_out([self.variable])
        print(f"WeekdayOneHotEncoder.fit")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        encoded_weekday = self.encoder.transform(X[[self.variable]])
        X[self.encoded_features_names] = encoded_weekday
        X.drop(self.variable,axis=1,inplace=True)
        print(f"WeekdayOneHotEncoder.transform")
        return X

