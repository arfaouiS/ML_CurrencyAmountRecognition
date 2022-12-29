from sklearn.model_selection import train_test_split
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
    dataframe["prewitt"] = dataframe["prewitt"].apply(lambda img_array: img_array / np.linalg.norm(img_array))
    dataframe["farid"] = dataframe["farid"].apply(lambda img_array: img_array / np.linalg.norm(img_array))
    dataframe["variance"] = dataframe["variance"].apply(lambda img_array: img_array / np.linalg.norm(img_array))
    dataframe["invert"] = dataframe["invert"].apply(lambda img_array: img_array / np.linalg.norm(img_array))
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
    dataframe["prewitt"] = dataframe["prewitt"].apply(lambda img_array: img_array.reshape(-1))
    dataframe["farid"] = dataframe["farid"].apply(lambda img_array: img_array.reshape(-1))
    dataframe["variance"] = dataframe["variance"].apply(lambda img_array: img_array.reshape(-1))
    dataframe["edge"] = dataframe["edge"].apply(lambda img_array: img_array.reshape(-1))
    dataframe["invert"] = dataframe["invert"].apply(lambda img_array: img_array.reshape(-1))
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
    dataframe["text"] = dataframe["text"].apply(lambda txt: int.from_bytes(txt.encode('unicode_escape'), "big"))
    return dataframe

'''
Preparation of data in a format acceptable by models and separation of data into training and test sets.
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
    X = dataframe.apply(lambda x: np.array(np.concatenate((x.image, x.gray_image, x.prewitt, x.farid, x.variance, x.edge, x.invert, x.sobel, x.gradient))), axis = 1)
    for i in range(len(X)):
        features.append(X[i])
    currency_values = dataframe.currency.values.reshape(-1, 1)
    amount_values = dataframe.amount.values.reshape(-1, 1)
    labels = np.concatenate([currency_values, amount_values], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels, test_size=0.2)
    return X_train, X_test, y_train, y_test


