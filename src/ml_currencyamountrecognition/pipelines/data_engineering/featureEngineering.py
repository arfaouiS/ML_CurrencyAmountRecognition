'''
     Convertir les images aux mêmes format  dans le dossier 02_intermediate
     Etudier les tailles et tout mettre à la même dimension
     Etudier la distribution des images dans un graphique
     Mettre dans un dataframe
        - Data augmentation
        - Feature extraction
'''


from PIL import Image
import pandas as pd
import numpy as np
import math
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

