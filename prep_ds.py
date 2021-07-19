import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def data_prep(data_dict, tokenizer, max_length, vocab_size, base_path, ds_path):
    X, y = list(), list()

    # For each image and list of captions
    for image_name, captions in data_dict.items():
        image_name = os.path.join(base_path, ds_path, image_name)

    # For each caption in the list of captions
    for caption in captions:

        # Convert the caption words into a list of word indices
        word_idxs = tokenizer.texts_to_sequences([caption])[0]

        # Pad the input text to the same fixed length
        # pad_idxs = pad_text(word_idxs, max_length)
        pad_idxs = pad_sequences([word_idxs], maxlen=max_length, padding='post')[0]
            
        X.append(image_name)
        y.append(pad_idxs)

    return np.array(X), np.array(y)
    return X, y

#Utility Function for replacing last term (if required)
def replace_last(source_string, replace_what, replace_with):
    head, _sep, tail = source_string.rpartition(replace_what)
    return head + replace_with + tail

# Load the numpy files
def map_func(img_name, cap):
    print(type(img_name))
    #Avoid decoding for now
    # img_tensor = np.load(img_name.decode('utf-8').replace(ds_path, feature_path, 1) + '.npy')
    img_tensor = np.load(img_name.replace(ds_path, feature_path, 1) + '.npy')
    return img_tensor, cap

def lazy_load_ds(train_X, train_y, BUFFER_SIZE, BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32]),num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset