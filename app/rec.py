import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import os

class RetailRec():
    # Pass dataframe as parameter and convert to dataset object
    def dataframe_to_dataset(self, df):
        df = df.copy()

        labels = df.pop("NEWDEPTSUBGROUP")
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
        ds = ds.shuffle(buffer_size=len(df))

        return ds

    def train(self):
        # Read csv data file and create pandas dataframe
        os.chdir('../data')

        df = pd.read_csv('transactions.csv')
        df.columns = map(lambda x: str(x).upper(), df.columns)
        df = df.drop(columns=['TRANSACTIONDATE'])
        df['ACTSALEVALUE'] = df['ACTSALEVALUE'].astype(int)
        print(df.shape)
        print(df.head())

        val_dataframe = df.sample(frac=0.2, random_state=1337)
        train_dataframe = df.drop(val_dataframe.index)
        print("Using %d samples for training and %d samples for validation" %(len(train_dataframe), len(val_dataframe)))

        # Convert the validation and training dataframes into datasets
        train_ds = self.dataframe_to_dataset(train_dataframe)
        val_ds = self.dataframe_to_dataset(val_dataframe)

        # Batch the datasets
        # train_ds = train_ds.batch(32)
        # val_ds = val_ds.batch(32)

        # CUSTOMERID,GenderDesc,MarketSectorDesc,AGE,AGEBAND,NEWDEPARTMENTDESC,NEWDEPTGROUP,NEWDEPTSUBGROUP,ACTSALEVALUE,TRANSACTIONDATE
        # The following features are categorical features encoded as integers
        # GenderDesc


        # The following features are continous numerical features
        # Age, ActSaleValue


        return "Done", 204

def main():
    rr = RetailRec()
    rr.train()

# Call
if __name__ == "__main__":
    main()