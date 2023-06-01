# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

loaded_model = load_model("face_detection.h5")

# Load an image
image_path = "C://Users//mak44/Desktop/real_and_fake_face/training_fake/hard_171_1101.jpg"
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))

# Preprocess the image
input_image = tf.keras.preprocessing.image.img_to_array(image)
input_image = np.expand_dims(input_image, axis=0)
input_image = input_image / 255.0  # Rescale pixel values

# Make predictions
predictions = loaded_model.predict(input_image)

# Get the predicted class label
predicted_class = "Fake" if predictions[0] < 0.5 else "Real"
confidence = predictions[0] if predictions[0] < 0.5 else 1 - predictions[0]

print("Predicted Class:", predicted_class)
print("Confidence:", confidence)