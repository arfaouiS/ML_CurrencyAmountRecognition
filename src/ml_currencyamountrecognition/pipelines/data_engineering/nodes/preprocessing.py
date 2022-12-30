from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
import pandas as pd
import numpy as np

'''
Translate all numerical values of image arrays into the range [0, 1].
Args:
    dataframe: dataframe containing all features and labels
Returns
    dataframe : normalized dataframe 
'''
def data_normalisation(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["image"] = dataframe["image"].apply(lambda img_array: img_array / np.linalg.norm(img_array))
    dataframe["gray_image"] = dataframe["gray_image"].apply(lambda img_array: img_array / np.linalg.norm(img_array))
    dataframe["variance"] = dataframe["variance"].apply(lambda img_array: img_array / np.linalg.norm(img_array))
    dataframe["sobel"] = dataframe["sobel"].apply(lambda img_array: img_array / np.linalg.norm(img_array))
    dataframe["gradient"] = dataframe["gradient"].apply(lambda img_array: img_array / np.linalg.norm(img_array))
    return dataframe

'''
Puts the data to the same dimensions, all arrays of more than 1 dimensions are converted into 1D-array.
Args :
    dataframe: dataframe containing all features and labels
Returns
    dataframe : standardized dataframe 
'''
def data_standardisation(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["image"] = dataframe["image"].apply(lambda img_array: img_array.reshape(-1))
    dataframe["gray_image"] = dataframe["gray_image"].apply(lambda img_array: img_array.reshape(-1))
    dataframe["variance"] = dataframe["variance"].apply(lambda img_array: img_array.reshape(-1))
    dataframe["edge"] = dataframe["edge"].apply(lambda img_array: img_array.reshape(-1))
    dataframe["sobel"] = dataframe["sobel"].apply(lambda img_array: img_array.reshape(-1))
    dataframe["gradient"] = dataframe["gradient"].apply(lambda img_array: img_array.reshape(-1))
    return dataframe

'''
Encodes ordinal data into numeric data for processing by models.
Args :
    dataframe: dataframe containing all features and labels
Returns
    dataframe : encoded dataframe 
'''
def ordinal_data_encoding(dataframe: pd.DataFrame) -> pd.DataFrame:
    label_encoder = preprocessing.LabelEncoder()
    dataframe["currency"] = label_encoder.fit_transform(dataframe["currency"])
    labels = dataframe.apply(lambda x: str(x.currency) + '-' + str(x.amount), axis=1)
    dataframe["labels"] = preprocessing.LabelEncoder().fit_transform(labels)
    dataframe["text"] = dataframe["text"].apply(lambda txt: int.from_bytes(txt.encode('unicode_escape'), "big"))
    dataframe['text'] = dataframe["text"].apply(lambda i: [float(i)])
    return dataframe


'''
Preparation of data in a format acceptable by Deep Learning models and separation of data into training and test sets.
Args:
    dataframe
Returns:
    X_train: train set of features
    X_test: test set of features
    y_train: train set of labels
    y_test: test set of labels
'''
def data_preparation_for_DeepLearning_models(dataframe: pd.DataFrame) -> pd.DataFrame:
    array_columns = ['image','gray_image','edge','variance','sobel','gradient','text']
    label_columns = ['currency','amount']
    max_length = dataframe[array_columns].apply(len).max()
    padded_arrays = []
    for i, row in dataframe.iterrows():
        arrays = []
        for array_column in array_columns:
            arrays.append(np.array(row[array_column]))
        padded_array = pad_sequences(arrays, maxlen=max_length)
        padded_arrays.append(padded_array)
    currency_values = dataframe[label_columns[0]].values.reshape(-1, 1)
    amount_values = (dataframe[label_columns[1]].astype(int)).values.reshape(-1, 1)
    labels = np.concatenate([currency_values, amount_values], axis=1)
    labels = labels.reshape(-1, len(label_columns))    
    X = np.array(padded_arrays)    
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X_train, X_test, X_val, y_train, y_test, y_val


'''
Preparation of data in a format acceptable by models other than Deep Learning ones and separation of data into training and test sets.
Args:
    dataframe
Returns:
    X_train: train set of features
    X_test: test set of features
    y_train: train set of labels
    y_test: test set of labels
'''

def data_preparation_for_MLmodels(dataframe: pd.DataFrame) -> pd.DataFrame:
    features = []
    X = dataframe.apply(lambda x: np.array(np.concatenate((x.image, x.gray_image, x.variance, x.sobel))), axis = 1)
    for i in range(len(X)):
        features.append(X[i])
    labels = dataframe.labels
    X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels, test_size=0.2)
    return X_train, X_test, y_train, y_test


