import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle


image_dir = '../data/Flickr8k_Dataset/'
output_feature_path = '../data/image_features.pkl'

# --------- Load Pretrained InceptionV3 ---------
model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

# --------- Feature Extraction Function ---------
def extract_feature(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature = model(img_array) 
    return feature[0].numpy()   

# --------- Process All Images ---------
features = {}
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
for idx, image_file in enumerate(image_files):
    path = os.path.join(image_dir, image_file)
    features[image_file] = extract_feature(path)
    if idx % 100 == 0:
        print(f'Processed {idx}/{len(image_files)} images')

# --------- Save Features ---------
with open(output_feature_path, 'wb') as f:
    pickle.dump(features, f)

print(f'Feature extraction complete. Saved to {output_feature_path}')
