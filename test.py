import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# Load the pretrained DenseNet121 model trained on ImageNet
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
num_classes = 10  # You should adjust this based on your dataset
predictions = Dense(num_classes, activation='softmax')(x)  # Adjust num_classes based on your dataset

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Load and preprocess the X-ray image you want to predict
img_path = 'x-ray2.jpg'  # Replace with the path to your X-ray image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)  # Preprocess the input image

# Make predictions
predictions = model.predict(img_array)

# Print the predictions
# Define your class labels or categories
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

# Get the index of the class with the highest probability
predicted_class_index = np.argmax(predictions)

# Get the predicted class label and its corresponding probability
predicted_class_label = class_labels[predicted_class_index]
predicted_probability = predictions[0][predicted_class_index]

# Print the predicted class label and probability
print("Predicted Disease:", predicted_class_label)
print("Probability:", predicted_probability)



# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# # Load the VGG16 model pretrained on ImageNet data
# model = VGG16(weights='imagenet', include_top=True)

# # Load and preprocess an image
# img_path = 'diease2.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# # Predict the class probabilities for the image
# preds = model.predict(x)
# # Decode and print the top-3 predicted classes
# decoded_preds = decode_predictions(preds, top=3)[0]

# # Filter out non-disease predictions
# disease_predictions = []
# for pred in decoded_preds:
#     if pred[0].startswith('n0'):  # ImageNet class IDs for diseases typically start with 'n0'
#         disease_predictions.append(pred)

# # Check if any disease predictions were found
# if len(disease_predictions) > 0:
#     print('Predicted Diseases:')
#     for pred in disease_predictions:
#         print(pred[1], '-', pred[2])
# else:
#     print('No disease predictions found.')
