from sklearn import preprocessing
import pandas as pd
import numpy as np


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


def ordinal_data_encoding(dataframe: pd.DataFrame) -> pd.DataFrame:
    label_encoder = preprocessing.LabelEncoder()
    onehot_encoder = preprocessing.OneHotEncoder()
    dataframe["currency"] = label_encoder.fit_transform(dataframe["currency"])
    dataframe["edge"] = dataframe["edge"].apply(lambda img_array: onehot_encoder.fit_transform(img_array).toarray())
    #dataframe["text"] = dataframe["text"].apply(lambda txt: txt.encode('unicode_escape'))
    return dataframe

