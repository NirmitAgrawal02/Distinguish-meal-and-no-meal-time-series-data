import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.fftpack import fft, ifft, rfft
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, RepeatedKFold
from joblib import dump, load
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

df_insulin_1 = pd.read_csv('InsulinData.csv', low_memory=False, usecols=[
    'Date', 'Time', 'BWZ Carb Input (grams)'])
df_cgm_1 = pd.read_csv('CGMData.csv', low_memory=False, usecols=[
    'Date', 'Time', 'Sensor Glucose (mg/dL)'])


df_insulin_1['date_time_stamp'] = pd.to_datetime(
    df_insulin_1['Date'] + ' ' + df_insulin_1['Time'])
df_cgm_1['date_time_stamp'] = pd.to_datetime(
    df_cgm_1['Date'] + ' ' + df_cgm_1['Time'])

df_insulin_2 = pd.read_csv('Insulin_patient2.csv', low_memory=False, usecols=[
    'Date', 'Time', 'BWZ Carb Input (grams)'])
df_cgm_2 = pd.read_csv('CGM_patient2.csv', low_memory=False, usecols=[
    'Date', 'Time', 'Sensor Glucose (mg/dL)'])
df_insulin_2['date_time_stamp'] = pd.to_datetime(
    df_insulin_2['Date'] + ' ' + df_insulin_2['Time'])
df_cgm_2['date_time_stamp'] = pd.to_datetime(
    df_cgm_2['Date'] + ' ' + df_cgm_2['Time'])


def mealDataExtraction(df_insulin, df_cgm, date_identifier):
    df_insulin = df_insulin.set_index('date_time_stamp')
    Valid_timestamps = []
    find_carb = df_insulin.sort_values(
        by='date_time_stamp', ascending=True).dropna().reset_index()
    find_carb['BWZ Carb Input (grams)'].replace(
        0.0, np.nan, inplace=True)
    find_carb = find_carb.dropna().reset_index().drop(columns='index')

    for i, timestamp in enumerate(find_carb['date_time_stamp']):
        try:
            time_difference = (
                find_carb['date_time_stamp'][i+1] - timestamp).seconds / 60.0
            if find_carb.loc[i, 'BWZ Carb Input (grams)'] > 0 and time_difference >= 120:
                Valid_timestamps.append(timestamp)
        except KeyError:
            pass

    meal_data = []
    if date_identifier == 1:
        for timestamp in Valid_timestamps:
            ts_start = pd.to_datetime(timestamp - timedelta(minutes=30))
            ts_end = pd.to_datetime(timestamp + timedelta(minutes=120))
            date_string = timestamp.date().strftime('%#m/%#d/%Y')
            meal_data.append(df_cgm.loc[df_cgm['Date'] == date_string].set_index('date_time_stamp').between_time(
                start_time=ts_start.strftime('%#H:%#M:%#S'), end_time=ts_end.strftime('%#H:%#M:%#S'))['Sensor Glucose (mg/dL)'].values.tolist())
        return pd.DataFrame(meal_data)
    else:
        for timestamp in Valid_timestamps:
            ts_start = pd.to_datetime(timestamp - timedelta(minutes=30))
            ts_end = pd.to_datetime(timestamp + timedelta(minutes=120))
            date_string = timestamp.date().strftime('%Y-%m-%d')
            meal_data.append(df_cgm.loc[df_cgm['Date'] == date_string].set_index('date_time_stamp').between_time(
                start_time=ts_start.strftime('%H:%M:%S'), end_time=ts_end.strftime('%H:%M:%S'))['Sensor Glucose (mg/dL)'].values.tolist())
        return pd.DataFrame(meal_data)


meal_data1 = mealDataExtraction(df_insulin_1, df_cgm_1, 1)
meal_data2 = mealDataExtraction(df_insulin_2, df_cgm_2, 2)
meal_data1 = meal_data1.iloc[:, 0:30]
meal_data2 = meal_data2.iloc[:, 0:30]


def noMealDataExtraction(df_insulin_data, df_cgm_data):
    insulin_no_meal = df_insulin_data.copy()
    insulin_no_meal = insulin_no_meal.sort_values(
        by='date_time_stamp', ascending=True).replace(0.0, np.nan).dropna().copy()
    insulin_no_meal = insulin_no_meal.reset_index().drop(columns='index')
    valid_timestamps = []
    for i, timestamp in enumerate(insulin_no_meal['date_time_stamp']):
        try:
            time_diff = (
                insulin_no_meal['date_time_stamp'][i+1]-timestamp).seconds // 3600
            if time_diff >= 4:
                valid_timestamps.append(timestamp)
        except KeyError:
            pass

    no_meal_data = []
    for i, timestamp in enumerate(valid_timestamps):
        counter = 1
        try:
            length = len(df_cgm_data.loc[(df_cgm_data['date_time_stamp'] >= valid_timestamps[i]+pd.Timedelta(
                minutes=120)) & (df_cgm_data['date_time_stamp'] < valid_timestamps[i+1])]) // 24
            while counter <= length:
                if counter == 1:
                    no_meal_data.append(df_cgm_data.loc[(df_cgm_data['date_time_stamp'] >= valid_timestamps[i]+pd.Timedelta(minutes=120)) & (
                        df_cgm_data['date_time_stamp'] < valid_timestamps[i+1])]['Sensor Glucose (mg/dL)'][:counter*24].values.tolist())
                    counter += 1
                else:
                    no_meal_data.append(df_cgm_data.loc[(df_cgm_data['date_time_stamp'] >= valid_timestamps[i]+pd.Timedelta(minutes=120)) & (
                        df_cgm_data['date_time_stamp'] < valid_timestamps[i+1])]['Sensor Glucose (mg/dL)'][(counter-1)*24:(counter)*24].values.tolist())
                    counter += 1
        except IndexError:
            break
    return pd.DataFrame(no_meal_data)


patient_no_meal_1 = noMealDataExtraction(df_insulin_1, df_cgm_1)
patient_no_meal_2 = noMealDataExtraction(df_insulin_2, df_cgm_2)


def creating_meal_feature_matrix(meal_data):
    preprocessed_data = meal_data.drop(meal_data.isna().sum(axis=1).replace(0, np.nan).dropna(
    ).where(lambda x: x > 6).dropna().index).reset_index().drop(columns='index')
    preprocessed_data = preprocessed_data.interpolate(method='linear', axis=1)
    index_to_drop = preprocessed_data.isna().sum(
        axis=1).replace(0, np.nan).dropna().index
    preprocessed_data = preprocessed_data.drop(
        meal_data.index[index_to_drop]).reset_index().drop(columns='index')
    tau_time = (preprocessed_data.iloc[:, 22:25].idxmin(
        axis=1) - preprocessed_data.iloc[:, 5:19].idxmax(axis=1)) * 5
    difference_in_glucose_normalized = (preprocessed_data.iloc[:, 5:19].max(
        axis=1) - preprocessed_data.iloc[:, 22:25].min(axis=1)) / (preprocessed_data.iloc[:, 22:25].min(axis=1))
    preprocessed_data = preprocessed_data.dropna().reset_index().drop(columns='index')
    power_at_first_max = []
    index_of_first_max = []
    power_at_second_max = []
    index_of_second_max = []
    for i in range(len(preprocessed_data)):
        fft_results = abs(
            rfft(preprocessed_data.iloc[:, 0:30].iloc[i].values.tolist())).tolist()
        sorted_fft_results = fft_results.copy()
        sorted_fft_results.sort()
        power_at_first_max.append(sorted_fft_results[-2])
        power_at_second_max.append(sorted_fft_results[-3])
        index_of_first_max.append(fft_results.index(sorted_fft_results[-2]))
        index_of_second_max.append(fft_results.index(sorted_fft_results[-3]))

    maximum = preprocessed_data.iloc[:, 5:19].idxmax(axis=1)
    tm = preprocessed_data.iloc[:, 22:25].idxmin(axis=1)
    first_differential = []
    second_differential = []
    for i in range(len(preprocessed_data)):
        data = preprocessed_data.iloc[:, maximum[i]:tm[i]].iloc[i].tolist()
        first_differential.append(np.diff(data).max())
        second_differential.append(np.diff(np.diff(data)).max())

    meal_feature_matrix = pd.DataFrame({
        'tau_time': tau_time,
        'difference_in_glucose_normalized': difference_in_glucose_normalized,
        'power_at_first_max': power_at_first_max,
        'power_at_second_max': power_at_second_max,
        'index_of_first_max': index_of_first_max,
        'index_of_second_max': index_of_second_max,
        'first_differential': first_differential,
        'second_differential': second_differential,
    })

    return meal_feature_matrix


meal_feature_matrix1 = creating_meal_feature_matrix(meal_data1)
meal_feature_matrix2 = creating_meal_feature_matrix(meal_data2)
meal_feature_matrix = pd.concat(
    [meal_feature_matrix1, meal_feature_matrix2]).reset_index().drop(columns='index')


def creating_no_meal_feature_matrix(non_meal_data):
    index_to_remove = non_meal_data.isna().sum(axis=1).replace(
        0, np.nan).dropna().where(lambda x: x > 5).dropna().index
    non_meal_data_cleaned = non_meal_data.drop(
        non_meal_data.index[index_to_remove]).reset_index().drop(columns='index')
    non_meal_data_cleaned = non_meal_data_cleaned.interpolate(
        method='linear', axis=1)
    index_to_drop_again = non_meal_data_cleaned.isna().sum(
        axis=1).replace(0, np.nan).dropna().index
    non_meal_data_cleaned = non_meal_data_cleaned.drop(
        non_meal_data_cleaned.index[index_to_drop_again]).reset_index().drop(columns='index')
    non_meal_data_cleaned['tau_time'] = (
        24 - non_meal_data_cleaned.iloc[:, 0:19].idxmax(axis=1)) * 5
    non_meal_data_cleaned['difference_in_glucose_normalized'] = (non_meal_data_cleaned.iloc[:, 0:19].max(
        axis=1) - non_meal_data_cleaned.iloc[:, 24]) / (non_meal_data_cleaned.iloc[:, 24])
    power_at_first_max = []
    index_of_first_max = []
    power_at_second_max = []
    index_of_second_max = []
    for i in range(len(non_meal_data_cleaned)):
        array = abs(
            rfft(non_meal_data_cleaned.iloc[:, 0:24].iloc[i].values.tolist())).tolist()
        sorted_array = abs(
            rfft(non_meal_data_cleaned.iloc[:, 0:24].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        power_at_first_max.append(sorted_array[-2])
        power_at_second_max.append(sorted_array[-3])
        index_of_first_max.append(array.index(sorted_array[-2]))
        index_of_second_max.append(array.index(sorted_array[-3]))
    first_differential = []
    second_differential = []
    for i in range(len(non_meal_data_cleaned)):
        first_differential.append(
            np.diff(non_meal_data_cleaned.iloc[:, 0:24].iloc[i].tolist()).max())
        second_differential.append(
            np.diff(np.diff(non_meal_data_cleaned.iloc[:, 0:24].iloc[i].tolist())).max())
    no_meal_feature_matrix = pd.DataFrame({
        'tau_time': non_meal_data_cleaned['tau_time'],
        'difference_in_glucose_normalized': non_meal_data_cleaned['difference_in_glucose_normalized'],
        'power_at_first_max': power_at_first_max,
        'power_at_second_max': power_at_second_max,
        'index_of_first_max': index_of_first_max,
        'index_of_second_max': index_of_second_max,
        'first_differential': first_differential,
        'second_differential': second_differential,
    })
    return no_meal_feature_matrix


non_meal_feature_matrix1 = creating_no_meal_feature_matrix(patient_no_meal_1)
non_meal_feature_matrix2 = creating_no_meal_feature_matrix(patient_no_meal_2)
non_meal_feature_matrix = pd.concat(
    [non_meal_feature_matrix1, non_meal_feature_matrix2]).reset_index().drop(columns='index')

meal_feature_matrix['label'] = 1
non_meal_feature_matrix['label'] = 0
total_data = pd.concat(
    [meal_feature_matrix, non_meal_feature_matrix]).reset_index().drop(columns='index')
dataset = shuffle(total_data, random_state=1).reset_index().drop(
    columns='index')
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
principaldata = dataset.drop(columns='label')
accuracy = []
precision = []
recall = []
f1score = []
model = DecisionTreeClassifier(criterion="entropy")
for train_index, test_index in kfold.split(principaldata):
    data_train, data_test, label_train, label_test = principaldata.loc[train_index], principaldata.loc[
        test_index], dataset.label.loc[train_index], dataset.label.loc[test_index]
    model.fit(data_train, label_train)
    predictions = model.predict(data_test)
    accuracy.append(accuracy_score(label_test, predictions)*100)
    precision.append(precision_score(label_test, predictions)*100)
    recall.append(recall_score(label_test, predictions)*100)
    f1score.append(f1_score(label_test, predictions)*100)
dump(model, 'DecisionTreeClassifier.pickle')
print("Accuracy:", np.mean(accuracy))
print("Precision:", np.mean(precision))
print("Recall:", np.mean(recall))
print("F1 Score:", np.mean(f1score))
