import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
from pathlib import Path

class DataSetLoader:
      def __init__(self,file_path):
          super(DataSetLoader,self).__init__()
          self.file_path=Path(file_path).resolve()
          self.scaler=MinMaxScaler()

      def read_data(self):
          if self.file_path.suffix==".csv":
             return pd.read_csv(self.file_path)
          elif self.file_path.suffix in [".xlsx",".xls"]:
             return pd.read_excel(self.file_path)
          
          raise ValueError(f"Unsupported file type for: {self.file_path}")
             

      def cleandata(self,df):
          # Check for null values
          if df.isnull().any(axis=1).sum()>int(0.8*len(df)):
             raise ValueError("More than 80% of rows contain missing values")

          elif df.isnull().any(axis=1).sum()<int(0.8*len(df)):
               cols_to_impute=[cols for cols in df.columns if cols.dtype in ["float","int"]]
               numeric_df=df[cols_to_impute]
               numeric_scaled_df=self.scaler.fit_transform(numeric_df) 
               imputer=KNNImputer(n_neighbors=3)
               imputed_values=imputer.fit_transform(numeric_scaled_df)
               df[cols_to_impute]=self.scaler.inverse_transform(imputed_values)
          #Check outliers
          return df
      