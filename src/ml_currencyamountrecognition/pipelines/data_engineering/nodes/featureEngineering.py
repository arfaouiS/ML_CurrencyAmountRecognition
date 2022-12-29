from skimage import filters, feature
from scipy import ndimage as nd
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


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
    df_blur['image'] = df_blur["image"].apply(lambda img: cv2.medianBlur(img, 5))
    frames = (dataframe, df_rotation45, df_rotation90, df_rotation130, df_fliph, df_flipv, df_blur)
    return pd.concat(frames, ignore_index=True)


def feature_extraction(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["gray_image"] = dataframe["image"].apply(lambda img_array: cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY))
    dataframe["edge"] = dataframe["gray_image"].apply(lambda img_array: feature.canny(img_array, sigma=3))
    dataframe["prewitt"] = dataframe["gray_image"].apply(lambda img_array: filters.prewitt(img_array))
    dataframe["farid"] = dataframe["gray_image"].apply(lambda img_array: filters.farid(img_array))
    dataframe["variance"] = dataframe["gray_image"].apply(
        lambda img_array: nd.generic_filter(img_array, np.var, size=3))
    dataframe["invert"] = dataframe["image"].apply(lambda img_array: cv2.bitwise_not(img_array))
    dataframe["sobel"] = dataframe["image"].apply(lambda img_array: filters.sobel(img_array))
    dataframe["gradient"] = dataframe["image"].apply(lambda img_array: cv2.Laplacian(img_array, cv2.CV_64F))
    #dataframe["text"] = dataframe["image"].apply(lambda img_array: extract_text_from_image(img_array))
    return dataframe


# Function that crops the image depending on the size given in paramters
def crop_image(img, y, x, h, w):
    cropped_image = img[y:y + h, x:x + w]
    return cropped_image


# Function that detects the contour of all items in the image And returns a list of cropped images based on the detected items
def detect_contours_of_all_elements(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cropped_images = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if h > 28:
                cropped_image = crop_image(img_array, y, x, h, w)
                cropped_images.append(cropped_image)
    return cropped_images


# Function that denoise the image given in parameter
def treat_And_filter(image):
    plt.style.use('seaborn')
    dst = cv2.fastNlMeansDenoisingColored(image, None, 11, 6, 7, 21)
    img_denoised = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2GRAY)
    return gray_img


# Function that reads the text in the given image
def read_text(image_path):
    img = cv2.imread(image_path)
    # Cvt to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Get binary-mask
    msk = cv2.inRange(hsv, np.array([0, 0, 175]), np.array([179, 255, 255]))
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dlt = cv2.dilate(msk, krn, iterations=1)
    thr = 255 - cv2.bitwise_and(dlt, msk)
    d = pytesseract.image_to_string(thr, config="--psm 10")
    return d


def extract_text_from_image(image):  # image_path
    images = detect_contours_of_all_elements(image)
    texts = ''
    count = 0
    for img in images:
        # 1. Denoise & filter image
        filtered_img = treat_And_filter(img)
        i = Image.fromarray(filtered_img)
        filtered_img_path = r'data\02_intermediate\temp_' + str(count) + '.jpg'
        count = count + 1
        i.save(filtered_img_path)
        # 2. Read text
        text = read_text(filtered_img_path)
        if (len(text[:-1]) != 0 and text[:-1] not in texts):
            texts = texts + text[:-1]
    return texts

