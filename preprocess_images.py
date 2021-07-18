
import os
import tensorflow as tf
import numpy as np
import math
from tensorflow.keras.models import Model
from tensorflow.keras.applications import inception_v3, InceptionV3
from tqdm import tqdm

from load_dataset import loadCaptions, getImages

def load_image(image_path):
  img = tf.io.read_file(image_path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.resize(img, (299, 299))
  img = inception_v3.preprocess_input(img)
  return img, image_path

def get_images(base_path, train_images_path, ds_path, feature_path):
  img_name_list = getImages(os.path.join(base_path, train_images_path))
  print(len(img_name_list))

  img_path_list = [os.path.join(base_path, ds_path, filename) for filename in img_name_list]
  print(img_path_list[0])
  img_dataset = tf.data.Dataset.from_tensor_slices(img_path_list)
  img_dataset = img_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(16)

  return img_name_list, img_path_list, img_dataset


def preprocess_images(img_name_list, img_path_list, img_dataset):
  image_model = InceptionV3(include_top=False, weights='imagenet')
  #Retreive and load images into tf.data.Dataset

  #Create Model
  new_input = image_model.input
  hidden_layer = image_model.layers[-1].output
  features_model = tf.keras.Model(new_input, hidden_layer)
  print(features_model.summary())

  for img, path in tqdm(img_dataset):
    batch_features = features_model(img)
    batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))

    for bf, p in zip(batch_features, path):
      path_of_feature = p.numpy().decode("utf-8").replace(ds_path, feature_path, 1)
      np.save(path_of_feature, bf.numpy())
    
  return