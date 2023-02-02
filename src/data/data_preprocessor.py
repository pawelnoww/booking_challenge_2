import copy
import os
import pickle

import numpy as np
import sklearn.preprocessing
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.utils.solutionutils import get_project_root
from pathlib import Path
from tqdm import tqdm


class DataPreprocessor:
    def __init__(self, df_path):
        self.df_path = df_path

        self.df = None

        self.encoder_city_id = None
        self.encoder_hotel_country = None
        self.encoder_device_class = None
        self.encoder_affiliate_id = None
        self.encoder_booker_country = None

        self.read_data()
        self.fill_nan_values()
        self.preprocess_data()
        self.sort_data()
        self.drop_cols()

    def read_data(self):
        self.df = pd.read_csv(self.df_path)

    def drop_cols(self):
        self.df.drop(['checkin', 'checkout'], axis=1, inplace=True)

    def sort_data(self):
        self.df = self.df.sort_values(by=['utrip_id', 'checkin'])

    def fill_nan_values(self):
        self.df.dropna(inplace=True)

    def preprocess_data(self):
        # Encode values
        self.encode_city()
        self.encode_hotel_country()
        self.encode_device_class()
        self.encode_affiliate_id()
        self.encode_booker_country()

        # One hot encoding
        self.one_hot_from_column('device_class')
        self.one_hot_from_column('booker_country')

        self.drop_short_trips()
        self.extract_time_features()

    def one_hot_from_column(self, column):
        one_hot = pd.get_dummies(self.df[column], prefix=column)
        self.df.drop(column, axis=1, inplace=True)
        self.df = self.df.join(one_hot)

    def encode_city(self):
        # N unique values of city_id: 39901
        encoder_city_id = LabelEncoder()
        self.df['city_id'] = encoder_city_id.fit_transform(self.df['city_id'])
        self.encoder_city_id = encoder_city_id

    def encode_hotel_country(self):
        # N unique values of hotel_country: 195
        encoder_hotel_country = LabelEncoder()
        self.df['hotel_country'] = encoder_hotel_country.fit_transform(self.df['hotel_country'])
        self.encoder_hotel_country = encoder_hotel_country

    def encode_device_class(self):
        # N unique values of device_class: 3
        encoder_device_class = LabelEncoder()
        self.df['device_class'] = encoder_device_class.fit_transform(self.df['device_class'])
        self.encoder_device_class = encoder_device_class

    def encode_affiliate_id(self):
        # N unique values of affiliate_id: 3254
        encoder_affiliate_id = LabelEncoder()
        self.df['affiliate_id'] = encoder_affiliate_id.fit_transform(self.df['affiliate_id'])
        self.encoder_affiliate_id = encoder_affiliate_id

    def encode_booker_country(self):
        # N unique values of booker_country: 5
        encoder_booker_country = LabelEncoder()
        self.df['booker_country'] = encoder_booker_country.fit_transform(self.df['booker_country'])
        self.encoder_booker_country = encoder_booker_country

    def drop_short_trips(self):
        # Drop rows with utrip_id length < 2
        self.df['Size'] = self.df.utrip_id.map(self.df.groupby('utrip_id').agg('size'))
        self.df = self.df[self.df.Size > 2]

    def extract_time_features(self):
        # Extract features from timestamp
        self.df['checkin'] = pd.to_datetime(self.df['checkin'])
        self.df['dayofweek_checkin'] = np.sin(self.df['checkin'].dt.dayofweek.values * np.pi / 7)
        self.df['dayofmonth_checkin'] = np.sin(self.df['checkin'].dt.day.values * np.pi / 30)
        self.df['month_checkin'] = np.sin(self.df['checkin'].dt.month.values * np.pi / 12)

        self.df['checkout'] = pd.to_datetime(self.df['checkout'])
        self.df['dayofweek_checkout'] = np.sin(self.df['checkout'].dt.dayofweek.values * np.pi / 7)
        self.df['dayofmonth_checkout'] = np.sin(self.df['checkout'].dt.day.values * np.pi / 30)
        self.df['month_checkout'] = np.sin(self.df['checkout'].dt.month.values * np.pi / 12)

    def get_list_of_sequences(self):
        X = []
        y = []
        features = []
        temp_sequence = []

        for index, row in tqdm(df.df.iterrows(), total=len(self.df)):
            if len(temp_sequence) == 0 or temp_sequence[-1]['utrip_id'] == row['utrip_id']:
                temp_sequence.append(row)
            else:
                X.append([x['city_id'] for x in temp_sequence[:-1]])
                y.append(temp_sequence[-1]['city_id'])
                features.append(temp_sequence[-1].values)
                temp_sequence = [row]

        X.append([x['city_id'] for x in temp_sequence[:-1]])
        y.append(temp_sequence[-1]['city_id'])
        features.append(temp_sequence[-1].values)

        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)

        return (X, y, features)

    def get_sequences_vectors(self):
        X, y, features = self.get_list_of_sequences()
        features = np.asarray(features)
        seq = tf.keras.preprocessing.sequence.pad_sequences(
            X,
            maxlen=10,
            dtype='int32',
            padding='post',
            truncating='post',
        )
        return np.hstack((seq, features)), np.asarray(y)


    @property
    def n_city(self):
        return self.df['city_id'].nunique()

    @property
    def n_hotel(self):
        return self.df['hotel_id'].nunique()

if __name__ == "__main__":

    # Zrobic embedding layer
    # layers.Embedding(n_cities, output_dim, n_samples)

    df = DataPreprocessor(str(Path(get_project_root() + "/data/train_set.csv")))


    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(256, input_shape=(30, 1), activation='tanh', return_sequences=True))
    model.add(tf.keras.layers.LSTM(128, activation='tanh',))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(df.n_city, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=4)])


    X, y = df.get_sequences_vectors()
    print(X.shape)
    X = np.expand_dims(X, -1)
    print(X.shape)
    # print(X, y)
    # print(df.get_n_cities())
    model.fit(X, y, epochs=100, batch_size=4096, validation_split=0.2)
