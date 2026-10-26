import os
import cv2 as cv
import numpy as np
import pandas as pd
import time as t
#import pyarrow as pa
#import pyarrow.parquet as pq
#import dask.array as da
#import dask.dataframe as dd
#from dask_ml.model_selection import train_test_split
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# Declares the locations of all cards and the number of each type.
TEMP_MTG = 'data/_temp/temp_mtg.dat'
TEMP_PKM = 'data/_temp/temp_pkm.dat'
TEMP_YGO = 'data/_temp/temp_ygo.dat'
TEMP_ALL = 'data/_temp/temp_all.dat'
PATH_MTG = 'data/mtg'
PATH_PKM = 'data/pokemon'
PATH_YGO = 'data/yugioh'
COUNT_MTG = 16_990
COUNT_PKM = 21_199
COUNT_YGO = 14_599
IMAGE_DATA_SIZE = 350_000


class CardType(Enum):
    """Card Type Enum for Magic, Pokemon, and Yu-Gi-Oh."""
    MTG = 0     # Magic: The Gathering
    PKM = 1     # Pokemon
    YGO = 2     # Yu-Gi-Oh


def convert_images_to_vectors(temporary_filename: str, path: str, count: int | None = None):
    """
    Converts the card JPGs in the path to Numpy normalized grayscale vectors.
    The vectors are normalize to [0, 1] using Min Max Scaling.

    Args:
        path (str): The data path for the card JPGs.
        count (int | None): The number of cards to process. All in path if None.
        temporary_filename (str): The temporary path/filename to store the card data on drive.
    
    Returns:
        A list of numpy 1D vectors for each image processed.
    """
    # Establish the number of cards to process
    start = t.perf_counter()
    filenames = os.listdir(path)
    if count is None: count = len(filenames)
    filenames = filenames[:count]
    card_vectors = np.memmap(temporary_filename, dtype=np.float32, mode='w+', shape=(count, IMAGE_DATA_SIZE))

    # Convert count number of cards
    for i, filename in enumerate(filenames):
        filepath = os.path.join(path, filename)
        card = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
        if card is not None:
            card = card.ravel()
            card = cv.normalize(src=card, dst=card_vectors[i], alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        else:
            print(f"[ERROR] Could not process card {filepath}.")
            card_vectors[i] = 0     # Fallback case

    # Flush the final bit of memory from RAM to the drive
    card_vectors.flush()

    # Returns all of the cards as Numpy grayscale vectors.
    end = t.perf_counter()
    print(f"\tTime Elapsed: {end - start}s.")
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
    start = t.perf_counter()
    card_counts = [len(card_list) for card_list in card_lists]
    label_data = np.repeat(card_labels, card_counts)   # Repeats each label count times

    # Combine all of the card vectors together
    card_data = np.memmap(TEMP_ALL, dtype=np.float32, mode='w+', shape=(sum(card_counts), IMAGE_DATA_SIZE))
    card_index = 0
    for card_count, card_list in zip(card_counts, card_lists):
        card_data[card_index : card_index + card_count, ...] = card_list
        card_index += card_count
    card_data.flush()
    end = t.perf_counter()
    print(f"\tTime Elapsed: {end - start}s.")

    """
    # Create the dataframe and filters out bad data
    start = t.perf_counter()
    card_data_dask_array = da.from_array(card_data, chunks=(sum(card_counts), IMAGE_DATA_SIZE))
    dataframe = dd.from_dask_array(card_data_dask_array, columns=[f"card_pxl_{i+1}" for i in range(IMAGE_DATA_SIZE)])
    label_dask_dask_array = da.from_array(label_data, chunks=(len(label_data),))
    dataframe['label'] = label_dask_dask_array
    valid_rows_mask = dataframe[dataframe.columns[1:]].any(axis=1)  # Creates a mask for rows: Not all 0s (True), all 0s (False)
    dataframe = dataframe[valid_rows_mask]                          # Removes any all 0 rows from the dataframe
    """

    # Create the dataframe and filters out bad data
    start = t.perf_counter()
    dataframe = pd.DataFrame(card_data, columns=[f"card_pxl_{i+1}" for i in range(IMAGE_DATA_SIZE)])
    dataframe.insert(0, 'label', label_data)
    valid_rows_mask = dataframe[dataframe.columns[1:]].any(axis=1)  # Creates a mask for rows: Not all 0s (True), all 0s (False)
    dataframe = dataframe[valid_rows_mask]                          # Removes any all 0 rows from the dataframe

    # Remove the file link and delete the temporary all file
    if hasattr(card_data, '_mmap'): card_data._mmap.close()
    del card_data
    if os.path.exists(TEMP_ALL): os.remove(TEMP_ALL)
    end = t.perf_counter()
    print(f"\tTime Elapsed: {end - start}s.")

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
    card_count = 100     # 10 for testing

    # Gets the vectors for all cards JPGs per path
    print("[INFO] Converting 'Magic: The Gathering' card JPGs to vectors...")
    mtg_cards = convert_images_to_vectors(TEMP_MTG, PATH_MTG, card_count)
    #mtg_cards = convert_images_to_vectors(TEMP_MTG, PATH_MTG, COUNT_MTG)
    print("[INFO] Converting 'Pokemon' card JPGs to vectors...")
    pkm_cards = convert_images_to_vectors(TEMP_PKM, PATH_PKM, card_count)
    #pkm_cards = convert_images_to_vectors(TEMP_PKM, PATH_PKM, COUNT_PKM)
    print("[INFO] Converting 'Yu-Gi-Oh' card JPGs to vectors...")
    ygo_cards = convert_images_to_vectors(TEMP_YGO, PATH_YGO, card_count)
    #ygo_cards = convert_images_to_vectors(TEMP_YGO, PATH_YGO, COUNT_YGO)

    # Add the card vectors into a Pandas dataframe
    print("[INFO] Creating Pandas dataframe...")
    card_lists = [mtg_cards, pkm_cards, ygo_cards]
    card_labels = ['MTG', 'PKM', 'YGO']
    df = create_dataframe(card_lists, card_labels)

    # Remove the file links and delete the temporary files
    if hasattr(mtg_cards, '_mmap'): mtg_cards._mmap.close()
    if hasattr(pkm_cards, '_mmap'): pkm_cards._mmap.close()
    if hasattr(ygo_cards, '_mmap'): ygo_cards._mmap.close()
    del mtg_cards, pkm_cards, ygo_cards
    if os.path.exists(TEMP_MTG): os.remove(TEMP_MTG)
    if os.path.exists(TEMP_PKM): os.remove(TEMP_PKM)
    if os.path.exists(TEMP_YGO): os.remove(TEMP_YGO)

    # Create the training, testing, and validation sets.
    # Train/Test is split 80/20 on total, and Train/Validation is split 80/20 on Train
    X = df.drop(columns=['label']).values
    y = df['label'].replace({
        'MTG': CardType.MTG.value,
        'PKM': CardType.PKM.value,
        'YGO': CardType.YGO.value
    }).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=35, stratify=y_train)

    # Balanced the class weighs since each set of cards has a different number of images
    classes = np.array([CardType.MTG.value, CardType.PKM.value, CardType.YGO.value])
    class_weights_array = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weights_dict = dict(zip(classes, class_weights_array))


if __name__ == "__main__":
    main()
