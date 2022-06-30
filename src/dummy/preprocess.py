import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from features import features_columns,  outcomes_columns, IS_XSRC, IS_YSRC, FID

def get_scaler(scaler):
    if scaler == 'standard':
        return StandardScaler()
    if scaler == 'robust':
        return RobustScaler()
    if scaler == 'minmax':
        return MinMaxScaler()
    raise ValueError()


class DataFrameImputer(TransformerMixin):
    def __init__(self):
        '''
        Impute missing values.
        Columns of dtype object are imputed with the most frequent value
        in column.
        Columns of other types are imputed with mean of column.
        '''

    def fit(self, X, y=None):
        fill = []
        for c in X:
            if X[c].dtype == np.dtype('bool'):
                fill.append(False)
            elif X[c].dtype == np.dtype('O'):
                fill.append(X[c].value_counts().index[0])
            else:
                fill.append(X[c].mean())
        self.fill = pd.Series(fill, index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

def  preprocess(data_file, winsor=99, scaler='standard', data_directory='./base',  input_missing=True, scaling=True):
    df = pd.read_csv(data_file)

    # replaces inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # drop columns if all the values are nan
    df.dropna(axis=1, how='all', inplace=True)

    float_features = [feature for (feature, dtype) in df.dtypes.iteritems() if dtype in (np.float32, np.float64) ]
    int_features = [feature for (feature, dtype) in df.dtypes.iteritems() if dtype in (np.int32, np.int64)]
    numeric_features = float_features + int_features

    # non_numeric_features = [(feature, dtype) for (feature, dtype) in df.dtypes.iteritems() if feature not in numeric_features]
    # print(non_numeric_features)

    # winsorize numeric features
    if winsor:
        for feature in numeric_features:
            p1, p2 = np.nanpercentile(df[feature], [winsor, 100-winsor])
            lower, upper = min(p1,p2), max(p1,p2)

            # select the feature column where df[feature] < lower
            df.loc[df[feature] < lower, feature] = lower
            df.loc[df[feature] > upper, feature] = upper

    # input missing values
    if input_missing:
        inputer = DataFrameImputer().fit(df[features_columns])
        df[features_columns] = inputer.transform(df[features_columns])

    inputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=False)
    inputer.fit(df[outcomes_columns])
    df[outcomes_columns] = inputer.transform(df[outcomes_columns])
    
    # scaling
    if scaling:
        scaler = get_scaler(scaler).fit(df[numeric_features])
        df[numeric_features] = scaler.transform(df[numeric_features])

    df['has_axis'] = df[[IS_XSRC,IS_YSRC]].apply(lambda x : x[IS_XSRC] or x[IS_YSRC], axis=1)
    df = df[df['has_axis']]
    df = df.drop('has_axis', axis=1)

    df['has_one_axis'] = df[[IS_XSRC,IS_YSRC]].apply(lambda x : not x[IS_XSRC] or not x[IS_YSRC], axis=1)
    df = df[df['has_one_axis']]
    df = df.drop('has_one_axis', axis=1)

    datasets = set(df[FID])
    datasets_alias = { ds : i for i, ds in enumerate(datasets) }
    df[FID] = df[FID].map(datasets_alias)

    return df


categories_id = { 
    ('bar', True, False) : 0,
    ('bar', False, True) : 1,
    # ('box', True, False) : 2,
    # ('box', False, True) : 2,
    # ('heatmap', True, False) : 3,
    # ('heatmap', False, True) : 4,
    # ('histogram', True, False) : 5,
    # ('histogram', False, True) : 5,
    ('line', True, False) : 2,
    ('line', False, True) : 3,
    ('scatter', True, False) : 4,
    ('scatter', False, True) : 5
}
