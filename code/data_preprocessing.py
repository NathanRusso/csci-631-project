import os

import cv2 as cv
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


"""
Returns card jpgs as a Numpy array of 3D BGR vectors.
"""
def convert_images_to_vectors(path, count):
    card_vectors = []

    for filename in os.listdir(path):
        if len(card_vectors) >= count:
            break

        filepath = os.path.join(path, filename)
        card = cv.imread(filepath)
        if card is not None:
            card_vectors.append(card)

    return card_vectors


"""
Converts to grayscale, normalizes image pixel values to 0-1 and returns a new array.
"""
def normalize_image_vectors(imgs):
    normalized_imgs = []

    for img in imgs:
        float_img = img.astype(np.float32)
        gray_image = cv.cvtColor(float_img, cv.COLOR_BGR2GRAY)
        norm_img = cv.normalize(gray_image, None, 0, 1, cv.NORM_MINMAX)
        normalized_imgs.append(norm_img)

    return normalized_imgs


"""
Creates a dictionary pairing cards with their brand label.
"""
def create_data_columns(brands, labels):
    label_data = []
    card_data = []
    for brand, label in zip(brands, labels):
        for card in brand:
            card.flatten()
            label_data.append(label)
            card_data.append(card)

    data = {
        'label': label_data,
        'card': card_data
    }

    return data


"""
Preprocessing of images:
1) Load images from folders
2) Encode as vectors
    3) Resize if necessary
    4) Color correction if necessary
    5) Noise removal if necessary
6) Normalize pixel values (0-1)
7) Create pandas dataframe with labels
8) Train/test/validation split with stratification
"""
def main():
    mtg_path = 'data/mtg'
    pkm_path = 'data/pokemon'
    ygo_path = 'data/yugioh'
    card_count = 10

    # jpg images --> vectors
    print("Converting images to vectors...")
    mtg_cards = convert_images_to_vectors(mtg_path, card_count)
    pkm_cards = convert_images_to_vectors(pkm_path, card_count)
    ygo_cards = convert_images_to_vectors(ygo_path, card_count)

    # resizing likely fine for now, all 500x700
    # not sure if we'll need color correction or noise removal

    # normalization (BGR values to 0-1)
    print("Normalizing pixel values...")
    mtg_cards = normalize_image_vectors(mtg_cards)
    pkm_cards = normalize_image_vectors(pkm_cards)
    ygo_cards = normalize_image_vectors(ygo_cards)

    # convert to pandas dataframe with labels
    print("Creating dataframe...")
    card_brands = [mtg_cards, pkm_cards, ygo_cards]
    card_labels = ['MTG', 'Pokemon', 'Yu-Gi-Oh']
    data = create_data_columns(card_brands, card_labels)
    df = pd.DataFrame(data)

    # train/test/validation split
    # train/test is 80/20, train/validation is also 80/20
    X = df['card']
    y = df['label'].replace({
        'MTG': 0, 
        'Pokemon': 1,
        'Yu-Gi-Oh': 2
        })

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=35, stratify=y_train)


if __name__ == "__main__":
    main()