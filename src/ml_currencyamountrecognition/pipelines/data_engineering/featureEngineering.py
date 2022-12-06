from skimage import filters, feature
from scipy import ndimage as nd
from PIL import Image
import pandas as pd
import numpy as np
import math
import cv2
import os


def banknote_dataframe(src: str, currencies: list) -> pd.DataFrame:
    pictures = []
    for currency in currencies:
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
