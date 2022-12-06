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


