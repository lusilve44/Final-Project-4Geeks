import pandas as pd
import numpy as np
import math
from scipy.stats import boxcox
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

X_train = pd.read_csv('../data/raw/X_train')
X_valid = pd.read_csv('../data/raw/X_valid')
X_test = pd.read_csv('../data/raw/X_test')

y_train = pd.read_csv('../data/raw/y_train')
y_valid = pd.read_csv('../data/raw/y_valid')
y_test = pd.read_csv('../data/raw/y_test')


def transformation_X(data):
    data['first_observation_date'] = pd.to_datetime(data['first_observation_date'])
    data['last_observation_date'] = pd.to_datetime(data['last_observation_date'])

    data['is_sentry_object'] = pd.Categorical(data['is_sentry_object'])
    data['is_sentry_object'] = data['is_sentry_object'].cat.codes

    data['orbit_class_type'] = pd.Categorical(data['orbit_class_type'])
    data['orbit_class_type'] = data['orbit_class_type'].cat.codes

    data['orbit_class_description'] = pd.Categorical(data['orbit_class_description'])
    data['orbit_class_description'] = data['orbit_class_description'].cat.codes

    data['name'] = pd.Categorical(data['name'])

    data['orbit_id'] = pd.to_numeric(data['orbit_id'])

    data['kilometers_estimated_diameter_min'] = data['kilometers_estimated_diameter_min']*1000
    data['kilometers_estimated_diameter_max'] = data['kilometers_estimated_diameter_max']*1000
    data.rename(columns = {'kilometers_estimated_diameter_min':'meters_estimated_diameter_min'}, inplace = True)
    data.rename(columns = {'kilometers_estimated_diameter_max':'meters_estimated_diameter_max'}, inplace = True)

    mean = (data['meters_estimated_diameter_max'] + data['meters_estimated_diameter_min'])/2
    data.drop(columns = ['meters_estimated_diameter_min'], inplace = True)
    data.rename(columns = {'meters_estimated_diameter_max':'meters_estimated_diameter'}, inplace = True)
    data['meters_estimated_diameter'] = mean

    data['volume (m^3)'] = (4/3)*math.pi*((data['meters_estimated_diameter_min']/2)**2)*(data['meters_estimated_diameter_max']/2)

    data['meters_estimated_diameter'], lambda1 = boxcox(data['meters_estimated_diameter'])
    data['perihelion_distance'], lambda2 = boxcox(data['perihelion_distance'])
    data['aphelion_distance'], lambda3 = boxcox(data['aphelion_distance'])
    data['volume (m^3)'], lambda4 = boxcox(data['volume (m^3)'])


def scaler_X(data):
    scaler = MinMaxScaler()
    scaler.fit(X_train[['absolute_magnitude_h', 'meters_estimated_diameter', 'perihelion_distance', 'aphelion_distance', 'volume (m^3)']])
    data[['absolute_magnitude_h', 'meters_estimated_diameter', 'perihelion_distance', 'aphelion_distance', 'volume (m^3)']] = scaler.transform(data[['absolute_magnitude_h', 'meters_estimated_diameter', 'perihelion_distance', 'aphelion_distance']])


def transformation_y(data):
    data['is_potentially_hazardous_asteroid'] = pd.Categorical(data['is_potentially_hazardous_asteroid'])
    data['is_potentially_hazardous_asteroid'] = data['is_potentially_hazardous_asteroid'].cat.codes


def imputer_train_valid(data):
    imputer_median1 = SimpleImputer(strategy='median', missing_values=np.nan)
    data['absolute_magnitude_h'] = imputer_median1.fit_transform(data[['absolute_magnitude_h']])

    imputer_median2 = SimpleImputer(strategy='median', missing_values=np.nan)
    data['meters_estimated_diameter'] = imputer_median2.fit_transform(data[['meters_estimated_diameter']])

    imputer_median3 = SimpleImputer(strategy='median', missing_values=np.nan)
    data['volume (m^3)'] = imputer_median3.fit_transform(data[['volume (m^3)']])





X_train = X_train.drop(X_train[X_train['orbit_id'] == 'E2021-CI3'].index[0])
X_train = X_train.drop(X_train[X_train['orbit_id'] == 'MPO392510'].index[0])
X_train['orbit_id'] = pd.to_numeric(X_train['orbit_id'])


# COSAS APARTE: fecha fea, lo de orbit_id