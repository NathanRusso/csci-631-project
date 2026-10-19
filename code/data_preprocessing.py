import os
import cv2 as cv
import pandas as pd
import numpy as np
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# Declares the locations of all cards and the number of each type.
PATH_MTG = 'data/mtg'
PATH_PKM = 'data/pokemon'
PATH_YGO = 'data/yugioh'
#COUNT_MTG = 20_000
COUNT_MTG = 16_991
COUNT_PKM = 21_199
COUNT_YGO = 14_599


class CardType(Enum):
    """Card Type Enum for Magic, Pokemon, and Yu-Gi-Oh."""
    MTG = 0     # Magic: The Gathering
    PKM = 1     # Pokemon
    YGO = 2     # Yu-Gi-Oh


def convert_images_to_vectors(path: str, count: int | None = None):
    """
    Converts the card JPGs in the path to Numpy normalized grayscale vectors.
    The vectors are normalize to [0, 1] using Min Max Scaling.

    Args:
        path (str): The data path for the card JPGs.
        count (int | None): The number of cards to process. All in path if None.
    
    Returns:
        A list of numpy 1D vectors for each image processed.
    """
    # Establish the number of cards to process
    card_vectors = []
    filenames = os.listdir(path)
    if count is None: count = len(filenames)
    filenames = filenames[:count]

    # Convert count number of cards
    for filename in filenames:
        filepath = os.path.join(path, filename)
        card = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
        if card is not None:
            card = card.ravel()
            card = cv.normalize(src=card, dst=None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            card_vectors.append(card)
        else:
            print(f"[ERROR] Could not process card {filepath}.")

    # Returns all of the cards as Numpy grayscale vectors.
    return card_vectors


def create_dataframe(card_lists: list, card_labels: list[str]):
    """
    Creates a dataframe mapping card vectors to their label.

    Args:
        card_lists (list): A list of list for each set of card vectors.
        card_labels (list[str]): A list of labels for each card type.

    Returns:
        A Pandas dataframe will all card mappings.
    """
    # Get the number of labels per card
    card_data = np.concatenate(card_lists, axis=0).tolist()
    card_counts = [len(card_list) for card_list in card_lists]
    label_data = np.repeat(card_labels, card_counts).tolist()   # Repeats each label count times

    # Set the data dictionary and convert it to a dataframe
    mapping = {'label': label_data, 'card': card_data}
    dataframe = pd.DataFrame(mapping)
    return dataframe


def main():
    """
    Prepares the data used to analyze the Magic: The Gathering, Pokemon, and Yu-Gi-Oh card images.

    Preprocessing of images:
    1) Convert image to vectors
        - Load images from folders
        - Encode them as vectors
        - Normalize the vectors [0, 1]
    2) Create a Pandas dataframe with labels
    3) Split the data into Train, Test, and Validation sets.
    """

    #card_count = min(COUNT_MTG, COUNT_PKM, COUNT_YGO)
    card_count = 10     # 10 for testing

    # Gets the vectors for all cards JPGs per path
    print("[INFO] Converting card JPGS to vectors...")
    mtg_cards = convert_images_to_vectors(PATH_MTG, card_count)
    pkm_cards = convert_images_to_vectors(PATH_PKM, card_count)
    ygo_cards = convert_images_to_vectors(PATH_YGO, card_count)

    # Add the card vectors into a Pandas dataframe
    print("[INFO] Creating Pandas dataframe...")
    card_lists = [mtg_cards, pkm_cards, ygo_cards]
    card_labels = ['MTG', 'PKM', 'YGO']
    df = create_dataframe(card_lists, card_labels)

    # Create the training, testing, and validation sets.
    # Train/Test is split 80/20 on total, and Train/Validation is split 80/20 on Train
    X = df['card']
    y = df['label'].replace({
        'MTG': CardType.MTG.value,
        'PKM': CardType.PKM.value,
        'YGO': CardType.YGO.value
    })
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=35, stratify=y_train)

    # Balanced the class weighs since each set of cards has a different number of images
    classes = np.array([CardType.MTG.value, CardType.PKM.value, CardType.YGO.value])
    class_weights_array = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weights_dict = dict(zip(classes, class_weights_array))


if __name__ == "__main__":
    main()
