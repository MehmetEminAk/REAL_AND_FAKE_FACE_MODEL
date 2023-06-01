import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Flatten , Dense , Dropout


train_dir = "C://Users/mak44/Desktop/splitted_dataset/train"
test_dir = "C://Users/mak44/Desktop/splitted_dataset/test"

model = Sequential()

model.add(Conv2D(32,(3,3),activation = "relu" , input_shape = (64,64,3)))
model.add(MaxPooling2D((3,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((3,3)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((3,3)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

#Compile the model


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load and preprocess your image dataset

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255 , shear_range=0.2,zoom_range=0.2)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)


train_generator = train_datagen.flow_from_directory(train_dir,target_size=(64,64),batch_size=32,class_mode="binary")
test_generator = train_datagen.flow_from_directory(test_dir,target_size=(64,64),batch_size=32,class_mode="binary")

# Train the model

model.fit(train_datagen,epochs = 10)

# Evaluate the model on test data
loss, accuracy = model.evaluate(test_generator, steps=len(test_generator))
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

#model.save("face_detection.h5")

