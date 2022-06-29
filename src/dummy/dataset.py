import pandas as pd
from .features import features_columns
import numpy as np

class Dataset:
    def __init__(self, dataset_file) -> None:
        self.dataset_file = dataset_file

        df = pd.read_csv(self.dataset_file)
        self.total_fods = max(df['fold'])

    
    def __iter__(self):
        for i in range(1, self.total_fods+1):
            df = pd.read_csv(self.dataset_file)
            fold_df = df[df['fold'] == i]
            del df # to save memory
            train_df = fold_df[fold_df['set'] == 'T'] 
            validate_df = fold_df[fold_df['set'] == 'V']
            del fold_df
            x_train, y_train = self.extract_x_y(train_df)
            x_validate, y_validate = self.extract_x_y(validate_df)
            del train_df
            del validate_df
            yield x_train, y_train, x_validate, y_validate

    def extract_x_y(self, df):
        return df[features_columns].to_numpy(dtype=np.float64), np.array(df['class'], dtype=np.int32)
