import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

train_mtg= np.memmap(
    "data/train/train_mtg.dat",
    dtype=np.float32,
    mode="r",
    shape=(9343,350_000),
)
val_mtg= np.memmap(
    "data/validation/validation_mtg.dat",
    dtype=np.float32,
    mode="r",
    shape=(2335,350_000),
)
test_mtg= np.memmap(
    "data/test/test_mtg.dat",
    dtype=np.float32,
    mode="r",
    shape=(2919,350_000),
)

train_pkm= np.memmap(
    "data/train/train_pkm.dat",
    dtype=np.float32,
    mode="r",
    shape=(9343,350_000),
)
val_pkm= np.memmap(
    "data/validation/validation_pkm.dat",
    dtype=np.float32,
    mode="r",
    shape=(2335,350_000),
)
test_pkm= np.memmap(
    "data/test/test_pkm.dat",
    dtype=np.float32,
    mode="r",
    shape=(2919,350_000),
)

train_ygo= np.memmap(
    "data/train/train_ygo.dat",
    dtype=np.float32,
    mode="r",
    shape=(9343,350_000),
)
val_ygo= np.memmap(
    "data/validation/validation_ygo.dat",
    dtype=np.float32,
    mode="r",
    shape=(2335,350_000),
)
test_ygo= np.memmap(
    "data/test/test_ygo.dat",
    dtype=np.float32,
    mode="r",
    shape=(2919,350_000),
)

def make_labeled_dataset(mtg, pkm, ygo, batch_size=32,shuffle= False ):
    """
    This function takes in an train/test/validation split from each of the card games, a batch size, and bool to determine 
    if we need to shuffle the dataset.
   
    It then processes them in batches in order to apply a label for each class so we can train the model with TensorFlow

    If prompted it can shuffle the dataset so the model can get a variety of classes instead of each one sequentially.
    This can reduce potentially reduce bias in the model


    It returns a tensorflow dataset that is processed from the from_generator function
    """

    
    arrays= [mtg,pkm,ygo]
    def batches():
        order = [
            (class_id, start)
            for class_id, data in enumerate(arrays)
            for start in range(0, len(data), batch_size)
        ]

        if shuffle:
            np.random.shuffle(order)

        for class_id, start in order:
            rows = np.asarray(arrays[class_id][start:start+batch_size])
            images = rows.reshape(-1, 700,500,1)
            labels = np.full(len(rows),class_id, dtype=np.int32)
            yield images, labels

    return tf.data.Dataset.from_generator(batches,
                                          output_signature=(
                                              tf.TensorSpec((None,700, 500, 1), tf.float32),
                                              tf.TensorSpec((None,), tf.int32)
                                          )).prefetch(tf.data.AUTOTUNE)

# size 32 batches proved to be the perfect balance between number of batches and time per batch.
# Average of 700ms per step with size 32 and anything more is 2 seconds or more per batch

train_dataset_labeled = make_labeled_dataset(train_mtg, train_pkm, train_ygo, batch_size=32, shuffle=True)

val_dataset_labeled = make_labeled_dataset( val_mtg, val_pkm, val_ygo, batch_size=32)

test_dataset_labled = make_labeled_dataset( test_mtg, test_pkm, test_ygo, batch_size=32)

train_steps = sum(
                (len(data)+ 31)
                for data in (train_mtg, train_pkm, train_ygo)
                )
val_steps = sum(
                (len(data)+ 31)
                for data in (val_mtg, val_pkm, val_ygo)
                )

model = keras.Sequential([
    keras.Input(shape=(700, 500, 1)),
    # we can further reduce image size which might save time but increase bias it is keras.layers.Resizing(175, 125),
    keras.layers.Conv2D(16, 3, activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(3,activation='softmax')
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(train_dataset_labeled, 
          epochs=1, 
          validation_data=val_dataset_labeled,
          shuffle = False, #did our own shuffling with numpy so turning this off to avoid a warning
          steps_per_epoch=train_steps,
          validation_steps=val_steps,)

model.evaluate(test_dataset_labled)