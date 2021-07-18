from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re


# In-place cleaning of captions
def captions_clean (image_dict):
  # <key> is the image_name, which can be ignored
  for key, captions in image_dict.items():
    
    # Loop through each caption for this image
    for i, caption in enumerate (captions):
      
      # Convert the caption to lowercase, and then remove all special characters from it
      caption_nopunct = re.sub(r"[^a-zA-Z0-9]+", ' ', caption.lower())
      
      # Remove all one-letter words
      caption_new = re.sub(r"\s[a-z]\s", ' ', caption_nopunct)
      
      # Replace the old caption in the captions list with this new cleaned caption
      captions[i] = caption_new


# Add two tokens, 'startseq' and 'endseq' at the beginning and end respectively
def add_token (captions):
  for i, caption in enumerate (captions):
    captions[i] = 'startseq ' + caption + ' endseq'
  return (captions)


# Get captions from images in image_dict that are present in image_names
def subset_data_dict (image_dict, image_names):
  # dict = { image_name:add_token(captions) for image_name,captions in image_dict.items() if image_name in image_names}
  dict = { image_name:captions for image_name,captions in image_dict.items() if image_name in image_names}
  return (dict)


# Get list of all captions
def all_captions (data_dict):
  return ([caption for key, captions in data_dict.items() for caption in captions])


def max_caption_length(captions):
  return max(len(caption.split()) for caption in captions)


def create_tokenizer(data_dict):
  captions = all_captions(data_dict)
  max_caption_words = max_caption_length(captions)
  
  # Initialise a Keras Tokenizer
  tokenizer = Tokenizer()
  
  # Fit it on the captions so that it prepares a vocabulary of all words
  tokenizer.fit_on_texts(captions)
  
  # Get the size of the vocabulary
  vocab_size = len(tokenizer.word_index) + 1

  return (tokenizer, vocab_size, max_caption_words)


def pad_text (text, max_length): 
  text = pad_sequences([text], maxlen=max_length, padding='post')[0]
  return (text)



