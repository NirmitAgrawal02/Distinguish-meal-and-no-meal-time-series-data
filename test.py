import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.fftpack import fft, ifft, rfft
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from joblib import dump, load
import pickle

data = pd.read_csv('test.csv', header=None)


def create_no_meal_feature_matrix(no_meal_data):
    no_meal_data['tau_time'] = (
        24 - no_meal_data.iloc[:, 0:19].idxmax(axis=1)) * 5
    no_meal_data['difference_in_glucose_normalized'] = (no_meal_data.iloc[:, 0:19].max(
        axis=1) - no_meal_data.iloc[:, 24]) / (no_meal_data.iloc[:, 24])
    first_power_max = []
    first_index_max = []
    second_power_max = []
    second_index_max = []
    for i in range(len(no_meal_data)):
        array = abs(
            rfft(no_meal_data.iloc[:, 0:24].iloc[i].values.tolist())).tolist()
        sorted_array = abs(
            rfft(no_meal_data.iloc[:, 0:24].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        first_power_max.append(sorted_array[-2])
        second_power_max.append(sorted_array[-3])
        first_index_max.append(array.index(sorted_array[-2]))
        second_index_max.append(array.index(sorted_array[-3]))
    first_differential = []
    second_differential = []
    for i in range(len(no_meal_data)):
        first_differential.append(
            np.diff(no_meal_data.iloc[:, 0:24].iloc[i].tolist()).max())
        second_differential.append(
            np.diff(np.diff(no_meal_data.iloc[:, 0:24].iloc[i].tolist())).max())
    no_meal_feature_matrix = pd.DataFrame({
        'tau_time': no_meal_data['tau_time'],
        'difference_in_glucose_normalized': no_meal_data['difference_in_glucose_normalized'],
        'power_at_first_max': first_power_max,
        'power_at_second_max': second_power_max,
        'index_of_first_max': first_index_max,
        'index_of_second_max': second_index_max,
        'first_differential': first_differential,
        'second_differential': second_differential,
    })
    return no_meal_feature_matrix


dataset = create_no_meal_feature_matrix(data)


with open('DecisionTreeClassifier.pickle', 'rb') as model:
    file = load(model)
    predictions = file.predict(dataset)
    model.close()

pd.DataFrame(predictions).to_csv('Result.csv', index=False, header=False)
