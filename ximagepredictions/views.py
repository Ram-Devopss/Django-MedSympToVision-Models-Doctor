from django.shortcuts import render
from django.http import HttpResponse

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import os


def x_ray(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage(location="./ximagepredictions/static/image")
        filename = fs.save(uploaded_file.name, uploaded_file)
        uploaded_file_url = fs.url(filename)

        # Path to the uploaded image
        img_path = os.path.join('./ximagepredictions/static/image', filename)
        

        # Load the pretrained DenseNet121 model trained on ImageNet
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Add custom classification layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        num_classes = 10  # Adjust this based on your dataset
        predictions = Dense(num_classes, activation='softmax')(x)  # Adjust based on your dataset

        # Create the model
        model = Model(inputs=base_model.input, outputs=predictions)


        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # Preprocess the input image


        # Make predictions
        predictions = model.predict(img_array)

        # Get the index of the class with the highest probability
        predicted_class_index = np.argmax(predictions)

        # Define disease labels
        class_labels = [
    "Pneumonia", "Bronchitis", "Asthma", "Tuberculosis", "Lung Cancer",
    "Pulmonary Edema", "COPD", "Emphysema", "Pneumothorax", "Fibrosis",
    "Cystic Fibrosis", "Pulmonary Embolism", "Lung Abscess", "Mesothelioma", 
    "Pulmonary Fibrosis", "Pulmonary Hypertension", "Sarcoidosis", "Hypersensitivity Pneumonitis",
    "Respiratory Syncytial Virus (RSV)", "Silicosis", "Asbestosis", "Pulmonary Alveolar Proteinosis", 
    "Smoke Inhalation Injury", "Eosinophilic Pneumonia", "ARDS", "Acute Bronchitis",
    "Alpha-1 Antitrypsin Deficiency", "Allergic Bronchopulmonary Aspergillosis (ABPA)",
    "Aspiration Pneumonia", "Atypical Pneumonia", "Black Lung Disease", "Bronchiectasis",
    "Bronchiolitis", "Chronic Cough", "Chronic Obstructive Pulmonary Disease (COPD)",
    "Chronic Respiratory Failure", "Collapsed Lung", "Diffuse Parenchymal Lung Disease",
    "Eosinophilic Granulomatosis with Polyangiitis (EGPA)", "Granulomatosis with Polyangiitis (GPA)",
    "Idiopathic Pulmonary Fibrosis (IPF)", "Interstitial Lung Disease (ILD)",
    "Langerhans Cell Histiocytosis (LCH)", "Pleural Effusion", "Pleural Mesothelioma",
    "Pulmonary Hypoplasia", "Pulmonary Langerhans Cell Histiocytosis (PLCH)",
    "Pulmonary Nodules", "Pulmonary Sarcoidosis", "Respiratory Failure", "Tuberculosis (TB)"
]

        # Get the predicted disease label and its corresponding probability
        predicted_disease = class_labels[predicted_class_index]
        probability = predictions[0][predicted_class_index]

  

        return render(request, 'result.html', {
            'uploaded_file_url': uploaded_file_url,
            'predicted_disease': predicted_disease,
            'probability': probability,
  
        })

    return render(request, 'index.html')
