import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import os
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

# Columns
# CUSTOMERID,GenderDesc,MarketSectorDesc,AGE,AGEBAND,NEWDEPARTMENTDESC,NEWDEPTGROUP,NEWDEPTSUBGROUP,ACTSALEVALUE,TRANSACTIONDATE


class RetailRec():
    def encode_numerical_feature(self, feature, name, dataset):
        # Create a Normalization layer for our feature
        normalizer = Normalization()

        # Prepare a Dataset that only yields our feature
        feature_ds = dataset.map(lambda x, y: x[name])
        feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

        # Learn the statistics of the data
        normalizer.adapt(feature_ds)

        # Normalize the input feature
        encoded_feature = normalizer(feature)

        return encoded_feature

    def encode_string_categorical_feature(self, feature, name, dataset):
        # Create a StringLookup layer which will turn strings into integer indices
        index = StringLookup()

        # Prepare a Dataset that only yields our feature
        feature_ds = dataset.map(lambda x, y: x[name])
        feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

        # Learn the set of possible string values and assign them a fixed integer index
        index.adapt(feature_ds)

        # Turn the string input into integer indices
        encoded_feature = index(feature)

        # Create a CategoryEncoding for our integer indices
        encoder = CategoryEncoding(output_mode="binary")

        # Prepare a dataset of indices
        feature_ds = feature_ds.map(index)

        # Learn the space of possible indices
        encoder.adapt(feature_ds)

        # Apply one-hot encoding to our indices
        encoded_feature = encoder(encoded_feature)

        return encoded_feature

    def encode_integer_categorical_feature(self, feature, name, dataset):
        # Create a CategoryEncoding for our integer indices
        encoder = CategoryEncoding(output_mode="binary")

        # Prepare a Dataset that only yields our feature
        feature_ds = dataset.map(lambda x, y: x[name])
        feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

        # Learn the space of possible indices
        encoder.adapt(feature_ds)

        # Apply one-hot encoding to our indices
        encoded_feature = encoder(feature)

        return encoded_feature

    def process_data(self):
        # Read csv data file and create pandas dataframe
        os.chdir('../data')
        df = pd.read_csv('transactions.csv')
        df.columns = map(lambda x: str(x).upper(), df.columns)
        df.sort_values(by=['CUSTOMERID', 'TRANSACTIONDATE'], ascending=True)

        # Group by the customer details and then transpose the most recent transactions
        df2 = df.groupby(['CUSTOMERID',
                          'GENDERDESC',
                          'MARKETSECTORDESC',
                          'AGE',
                          'AGEBAND', ])['NEWDEPTSUBGROUP'].apply(lambda df: df.reset_index(drop=True)).unstack().add_prefix('NEW_DEPT')

        # Get 15 most recent transactions
        df2 = df2[lambda df: df.columns[0:15]]
        df2 = df2.rename(columns={'NEW_DEPT14': "NEXTPURCHASE"})

        # Drop rows with null values
        df2 = df2.dropna()

        # df2.to_csv('transposed_transactions.csv')

        return df2

    # This function takes the dataframe and converts it to a dataset object
    def dataframe_to_dataset(self, df):
        df = df.copy()

        labels = df.pop("NEXTPURCHASE")
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
        ds = ds.shuffle(buffer_size=len(df))

        return ds

    # This function takes trains and builds the recommendation engine model
    def train(self, df2):
        # Set df = to pass value
        df = df2
        print(df)

        val_dataframe = df.sample(frac=0.2, random_state=1337)
        train_dataframe = df.drop(val_dataframe.index)
        print("Using %d samples for training and %d samples for validation" %
              (len(train_dataframe), len(val_dataframe)))

        # Convert the validation and training dataframes into datasets
        train_ds = self.dataframe_to_dataset(train_dataframe)
        val_ds = self.dataframe_to_dataset(val_dataframe)

        # Batch the datasets
        train_ds = train_ds.batch(32)
        val_ds = val_ds.batch(32)


        # Categorical feature encoded as string
        market_sector = keras.Input(shape=(1,), name="MARKETSECTORDESC", dtype="string")
        age_band = keras.Input(shape=(1,), name="AGEBAND", dtype="string")
        

        # Categorical features as encoded as integers
        gender = keras.Input(shape=(1,), name="GENDERDESC", dtype="int64")

        # Numerical features
        age = keras.Input(shape=(1,), name="AGE")
        # act_sale_value = keras.Input(shape=(1,), name="ACTSALEVALUE")

        # Categorical features encoded as string
        market_sector = keras.Input(shape=(1,), name="MARKETSECTORDESC", dtype="string")
        age_band = keras.Input(shape=(1,), name="AGEBAND", dtype="string")
        new_dept0 = keras.Input(shape=(1,), name='NEW_DEPT0', dtype="string")
        new_dept1 = keras.Input(shape=(1,), name='NEW_DEPT1', dtype="string")
        new_dept2 = keras.Input(shape=(1,), name='NEW_DEPT2', dtype="string")
        new_dept3 = keras.Input(shape=(1,), name='NEW_DEPT3', dtype="string")
        new_dept4 = keras.Input(shape=(1,), name='NEW_DEPT4', dtype="string")
        new_dept5 = keras.Input(shape=(1,), name='NEW_DEPT5', dtype="string")
        new_dept6 = keras.Input(shape=(1,), name='NEW_DEPT6', dtype="string")
        new_dept7 = keras.Input(shape=(1,), name='NEW_DEPT7', dtype="string")
        new_dept8 = keras.Input(shape=(1,), name='NEW_DEPT8', dtype="string")
        new_dept9 = keras.Input(shape=(1,), name='NEW_DEPT9', dtype="string")
        new_dept10 = keras.Input(shape=(1,), name='NEW_DEPT10', dtype="string")
        new_dept11 = keras.Input(shape=(1,), name='NEW_DEPT11', dtype="string")
        new_dept12 = keras.Input(shape=(1,), name='NEW_DEPT12', dtype="string")
        new_dept13 = keras.Input(shape=(1,), name='NEW_DEPT13', dtype="string")
        new_dept14 = keras.Input(shape=(1,), name='NEW_DEPT14', dtype="string")

        all_inputs = [
            gender,
            market_sector,
            age,
            age_band,
            new_dept0,
            new_dept1,
            new_dept2,
            new_dept3,
            new_dept4,
            new_dept5,
            new_dept6,
            new_dept7,
            new_dept8,
            new_dept9,
            new_dept10,
            new_dept11,
            new_dept12,
            new_dept13,
            new_dept14
        ]

        # Integer categorical features
        gender_encoded = self.encode_integer_categorical_feature(gender, "GENDERDESC", train_ds)

        # Numerical features
        age_encoded = self.encode_numerical_feature(age, "AGE", train_ds)

        # String categorical features
        market_sector_encoded = self.encode_string_categorical_feature(market_sector, "MARKETSECTORDESC", train_ds)
        age_band_encoded = self.encode_string_categorical_feature(age_band, "AGEBAND", train_ds)
        # new_dept0_encoded = encode_string_categorical_feature(, "", train_ds)
        # new_dept0_encoded = encode_string_categorical_feature(, "", train_ds)
        # new_dept0_encoded = encode_string_categorical_feature(, "", train_ds)
        # new_dept0_encoded = encode_string_categorical_feature(, "", train_ds)
        # new_dept0_encoded = encode_string_categorical_feature(, "", train_ds)
        # new_dept0_encoded = encode_string_categorical_feature(, "", train_ds)
        # new_dept0_encoded = encode_string_categorical_feature(, "", train_ds)
        # new_dept0_encoded = encode_string_categorical_feature(, "", train_ds)
        # new_dept0_encoded = encode_string_categorical_feature(, "", train_ds)
        # new_dept0_encoded = encode_string_categorical_feature(, "", train_ds)
        # new_dept0_encoded = encode_string_categorical_feature(, "", train_ds)
        # new_dept0_encoded = encode_string_categorical_feature(, "", train_ds)
        # new_dept0_encoded = encode_string_categorical_feature(, "", train_ds)
        # new_dept0_encoded = encode_string_categorical_feature(, "", train_ds)
        # new_dept0_encoded = encode_string_categorical_feature(, "", train_ds)


        return "Done", 204


def main():
    rr = RetailRec()
    df2 = rr.process_data()
    rr.train(df2)


# Call
if __name__ == "__main__":
    main()
