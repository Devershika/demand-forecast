import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
from pathlib import Path

class DataSetLoader:
      def __init__(self,file_path):
          super(DataSetLoader,self).__init__()
          project_root = Path.cwd()
          self.file_path = project_root/file_path
          self.scaler=MinMaxScaler()
      
      def read_data(self):
         if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

         if self.file_path.suffix==".csv":
            return pd.read_csv(self.file_path)

         if self.file_path.suffix in [".xlsx", ".xls"]:
            return pd.read_excel(self.file_path)

         raise ValueError(f"Unsupported file type: {self.file_path}")
             

      def cleandata(self,df):
          # Check for null values
          if df.isnull().any(axis=1).sum()>int(0.8*len(df)):
             raise ValueError("More than 80% of rows contain missing values")

          elif df.isnull().any(axis=1).sum()<int(0.8*len(df)):
               cols_to_impute=[cols for cols in df.columns if df[cols].dtype in ["float","int"]]
               numeric_df=df[cols_to_impute]
               numeric_scaled_df=self.scaler.fit_transform(numeric_df) 
               imputer=KNNImputer(n_neighbors=3)
               imputed_values=imputer.fit_transform(numeric_scaled_df)
               df[cols_to_impute]=self.scaler.inverse_transform(imputed_values)
               df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y",errors="coerce")
               df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
          #Check outliers
          return df
      