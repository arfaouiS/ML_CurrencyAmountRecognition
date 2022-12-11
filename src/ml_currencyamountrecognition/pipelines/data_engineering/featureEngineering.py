from skimage import filters, feature
from sklearn import preprocessing
from scipy import ndimage as nd
from PIL import Image
import pandas as pd
import numpy as np
import math
import cv2
import os


def banknote_dataframe(src: str, currency_folder_names: list) -> pd.DataFrame:
    pictures = []
    for currency in currency_folder_names:
        for filename in os.listdir(os.path.join(src, currency)):
            image_path = os.path.join(src, currency, filename)
            amount = filename.split('_')[1]
            img = Image.open(image_path).convert("RGB")
            pictures.append([np.asarray(img), currency, amount])
    return pd.DataFrame(pictures, columns=['image', 'currency', 'amount'])


def resize_pictures(dataframe: pd.DataFrame) -> pd.DataFrame:
    size_width, ratio = {}, []
    for i in dataframe.index:
        w, h, c = dataframe['image'][i].shape
        size_width[w * h] = w
        ratio.append(w / h)
    min_width = size_width[sorted(size_width.keys())[0]]
    width = int(math.ceil(min_width / 100.0)) * 100
    ratio = sum(ratio) / len(ratio)
    height = round(width * ratio)
    dataframe['image'] = dataframe['image'].apply(
        lambda img: np.asarray(Image.fromarray(np.uint8(img)).resize((width, height))))
    return dataframe


def img_rotation(img_array, rotation):
    height, width = img_array.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation, .5)
    rotated_img = cv2.warpAffine(img_array, rotation_matrix, (width, height))
    return rotated_img


def data_augmentation(dataframe: pd.DataFrame) -> pd.DataFrame:
    df_rotation45 = dataframe.copy()
    df_rotation90 = dataframe.copy()
    df_rotation130 = dataframe.copy()
    df_fliph = dataframe.copy()
    df_flipv = dataframe.copy()
    df_blur = dataframe.copy()
    df_rotation45['image'] = df_rotation45["image"].apply(lambda img: img_rotation(img, 45))
    df_rotation90['image'] = df_rotation90["image"].apply(lambda img: img_rotation(img, 90))
    df_rotation130['image'] = df_rotation130["image"].apply(lambda img: img_rotation(img, 140))
    df_fliph['image'] = df_fliph["image"].apply(lambda img: np.flipud(img))
    df_flipv['image'] = df_flipv["image"].apply(lambda img: np.fliplr(img))
    df_blur['image'] = df_blur["image"].apply(lambda img:cv2.medianBlur(img, 5))
    frames = (dataframe, df_rotation45, df_rotation90, df_rotation130, df_fliph, df_flipv, df_blur)
    return pd.concat(frames)


def feature_extraction(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["gray_image"] = dataframe["image"].apply(lambda img_array: cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY))
    dataframe["edge"] = dataframe["gray_image"].apply(lambda img_array: feature.canny(img_array))
    dataframe["prewitt_v"] = dataframe["gray_image"].apply(lambda img_array: filters.prewitt_v(img_array))
    dataframe["prewitt_h"] = dataframe["gray_image"].apply(lambda img_array: filters.prewitt_h(img_array))
    dataframe["farid"] = dataframe["gray_image"].apply(lambda img_array: filters.farid(img_array))
    dataframe["variance"] = dataframe["gray_image"].apply(lambda img_array: nd.generic_filter(img_array, np.var, size=3))
    dataframe["invert"] = dataframe["image"].apply(lambda img_array: cv2.bitwise_not(img_array))
    dataframe["sobel"] = dataframe["image"].apply(lambda img_array: filters.sobel(img_array))
    dataframe["gradient"] = dataframe["image"].apply(lambda img_array: cv2.Laplacian(img_array, cv2.CV_64F))
    return dataframe


def ordinal_data_encoding(dataframe: pd.DataFrame) -> pd.DataFrame:
    label_encoder = preprocessing.LabelEncoder()
    onehot_encoder = preprocessing.OneHotEncoder()
    dataframe["currency"] = label_encoder.fit_transform(dataframe["currency"])
    dataframe["edge"] = dataframe["edge"].apply(lambda img_array: onehot_encoder.fit_transform(img_array))
    return dataframe


def normalize_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["image"] = dataframe["image"].apply(lambda img_array: img_array/np.linalg.norm(img_array))
    dataframe["gray_image"] = dataframe["gray_image"].apply(lambda img_array: img_array/np.linalg.norm(img_array))
    dataframe["prewitt_h"] = dataframe["prewitt_h"].apply(lambda img_array: img_array/np.linalg.norm(img_array))
    dataframe["prewitt_v"] = dataframe["prewitt_v"].apply(lambda img_array: img_array/np.linalg.norm(img_array))
    dataframe["farid"] = dataframe["farid"].apply(lambda img_array: img_array/np.linalg.norm(img_array))
    dataframe["variance"] = dataframe["variance"].apply(lambda img_array: img_array/np.linalg.norm(img_array))
    dataframe["invert"] = dataframe["invert"].apply(lambda img_array: img_array/np.linalg.norm(img_array))
    dataframe["sobel"] = dataframe["sobel"].apply(lambda img_array: img_array/np.linalg.norm(img_array))
    dataframe["gradient"] = dataframe["gradient"].apply(lambda img_array: img_array/np.linalg.norm(img_array))
    return dataframe








