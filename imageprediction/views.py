from django.shortcuts import render
from django.http import HttpResponse

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def predict_it(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage(location="./imageprediction/static/image")
        filename = fs.save(uploaded_file.name, uploaded_file)
        uploaded_file_url = fs.url(filename)

        # Path to the uploaded image
        img_path = os.path.join('./imageprediction/static/image', filename)
        


        # Define constants
        IMAGE_WIDTH = 150
        IMAGE_HEIGHT = 150
        BATCH_SIZE = 32
        EPOCHS = 25
        NUM_CLASSES = 3  # Number of disease classes

        # Function to preprocess input image
        def preprocess_image(image_path):
            img = image.load_img(image_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize pixel values
            return img_array

        # Function to predict disease from input image
        def predict_disease(image_path, model):
            img_array = preprocess_image(image_path)
            predictions = model.predict(img_array)

            
            predicted_class_index = np.argmax(predictions)
            
            # Assuming your classes are ['cold', 'fever', 'headache']
            class_names = ['cold', 'fever', 'headache']
            predicted_class = class_names[predicted_class_index]
            
            return predicted_class

        # Data Augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )

        train_generator = train_datagen.flow_from_directory(
            './datasets',
            target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            './datasets',
            target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )

        # Define the CNN model
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(NUM_CLASSES, activation='softmax')
        ])

        # Hyperparameter Tuning
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        # Train the model
        model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps=len(validation_generator)
        )

        # Save the trained model
        model.save('./model/disease_prediction_model_v2.h5')

        # Load the trained model
        model = load_model('./model/disease_prediction_model_v2.h5')

        # Example usage
        input_image_path = img_path
        predicted_disease = predict_disease(input_image_path, model)
        print("Predicted disease:", predicted_disease)
        

        return render(request, 'result.html', {
            'uploaded_file_url': uploaded_file_url,
            'predicted_disease': predicted_disease,
  
            
  
        })

    return render(request, 'image.html')
