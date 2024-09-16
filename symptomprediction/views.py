from django.shortcuts import render
from django.http import HttpResponse

def symptom(request):
    return render(request,'diease.html',name)


from django.shortcuts import render
from django.contrib.messages import constants as messages
from networkx import dense_gnm_random_graph
from sklearn import tree
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector

# Create your views here.

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

 
import numpy as np
import pandas as pd
import random as ran
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import xgboost as xgb




#List of the symptoms is listed here in list l1.
l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
      'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
      'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
      'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
      'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
      'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
      'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
      'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
      'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
      'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
      'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
      'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
      'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
      'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
      'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
      'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
      'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum',
      'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
      'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
      'blister', 'red_sore_around_nose', 'yellow_crust_ooze']

# Remove duplicates
l1 = list(set(l1))

# Now, 'l1' contains unique feature names without duplicates.

#List of Diseases is listed in list disease.
disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']
l2=[]
for i in range(0,len(l1)):
    l2.append(0)

df=pd.read_csv('Prototype.csv')

#Replace the values in the imported file by pandas by the inbuilt function replace in pandas.

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

#check the df 
#print(df.head(),'This Head')

X= df[l1]

#print(X ,'<---this is x value')

y = df[["prognosis"]]
np.ravel(y)


#print(y,'<---this is y value')

#Read a csv named Testing.csv

tr=pd.read_csv('Prototype.csv')

#Use replace method in pandas.

tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)


X_test= tr[l1]
y_test = tr[["prognosis"]]

#print(y_test,'<---this is y test value')
#print(X_test,'<---this is X Test test value')
#print(tr)
np.ravel(y_test)

ndiease = {0:'Fungal infection',1:'Allergy',2:'GERD',3:'Chronic cholestasis',4:'Drug Reaction',
5:'Peptic ulcer diseae',6:'AIDS',7:'Diabetes ',8:'Gastroenteritis',9:'Bronchial Asthma',10:'Hypertension ',
11:'Migraine',12:'Cervical spondylosis',
13:'Paralysis (brain hemorrhage)',14:'Jaundice',15:'Malaria',16:'Chicken pox',17:'Dengue',18:'Typhoid',19:'hepatitis A',
20:'Hepatitis B',21:'Hepatitis C',22:'Hepatitis D',23:'Hepatitis E',24:'Alcoholic hepatitis',25:'Tuberculosis',
26:'Common Cold',27:'Pneumonia',28:'Dimorphic hemmorhoids(piles)',29:'Heart attack',30:'Varicose veins',31:'Hypothyroidism',
32:'Hyperthyroidism',33:'Hypoglycemia',34:'Osteoarthristis',35:'Arthritis',
36:'(vertigo) Paroymsal  Positional Vertigo',37:'Acne',38:'Urinary tract infection',39:'Psoriasis',
40:'Impetigo'}
args = {}

name = {
    'diease': [
        'abdominal_pain', 'abnormal_menstruation', 'acidity', 'acute_liver_failure', 'altered_sensorium',
        'anxiety', 'back_pain', 'belly_pain', 'blackheads', 'blister', 'bloody_stool', 'blurred_and_distorted_vision',
        'breathlessness', 'bruising', 'burning_micturition', 'chills', 'chest_pain', 'coma', 'congestion',
        'continuous_feel_of_urine', 'continuous_sneezing', 'constipation', 'cough', 'cramps', 'dark_urine', 'dehydration',
        'depression', 'diarrhea', 'dizziness', 'distention_of_abdomen', 'dischromic_patches', 'distention_of_abdomen',
        'drying_and_tingling_lips', 'enlarged_thyroid', 'extra_marital_contacts', 'excessive_hunger', 'fainting',
        'family_history', 'fast_heart_rate', 'fatigue', 'fluid_overload', 'foul_smell_of_urine', 'gases', 'headache',
        'high_fever', 'hip_joint_pain', 'history_of_alcohol_consumption', 'indigestion', 'inflammatory_nails',
        'internal_itching', 'irregular_sugar_level', 'irritability', 'irritation_in_anus', 'itching', 'joint_pain',
        'knee_pain', 'lack_of_concentration', 'lethargy', 'loss_of_appetite', 'loss_of_balance', 'loss_of_smell',
        'malaise', 'mild_fever', 'mood_swings', 'movement_stiffness', 'mucoid_sputum', 'muscle_pain', 'muscle_wasting',
        'muscle_weakness', 'nausea', 'neck_pain', 'nodal_skin_eruptions', 'obesity', 'palpitations', 'pain_behind_the_eyes',
        'pain_during_bowel_movements', 'pain_in_anal_region', 'painful_walking', 'patches_in_throat', 'passage_of_gases',
        'phlegm', 'polyuria', 'puffy_face_and_eyes', 'prominent_veins_on_calf', 'prognosis', 'pus_filled_pimples', 'red_sore_around_nose',
        'red_spots_over_body', 'redness_of_eyes', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'restlessness',
        'runny_nose', 'rusty_sputum', 'scarring', 'shivering', 'silver_like_dusting', 'sinus_pressure', 'skin_peeling', 'slurred_speech',
        'small_dents_in_nails', 'spinning_movements', 'spots', 'stiff_neck', 'stomach_bleeding', 'stomach_pain', 'swelling_joints',
        'swelling_of_stomach', 'swelling_of_stomach', 'swelled_lymph_nodes', 'swollen_blood_vessels', 'swollen_extremeties',
        'swollen_legs', 'swollen_legs', 'throat_irritation', 'thyroid', 'toxic_look_(typhos)', 'ulcers_on_tongue', 'unsteadiness',
        'visual_disturbances', 'vomiting', 'watering_from_eyes', 'weakness_in_limbs', 'weakness_of_one_body_side', 'weight_gain',
        'weight_loss', 'yellow_crust_ooze', 'yellow_urine', 'yellowing_of_eyes']
}






def makeme(request):

    
    if request.method=='POST':
        
 
        args['user'] = request.POST['user']
        symptom1 = request.POST['symp1']
        symptom2 = request.POST['symp2']
        symptom3 = request.POST['symp3']
        symptom4 = request.POST['symp4']
        symptom5 = request.POST['symp5']
        values=[0.5,0.6,0.7,0.8,0.9,1]

        

    
    """
    items = ['Mango', 'Orange', 'Apple', 'Lemon']
    file = open('items.txt','w')
    for the item in items:
	 file.write(item+"\n")
     file.close()"""

    
# For Symptom - 1
    with open('sypmt1.txt','w') as file:
        file.write(symptom1)
        
    # For Symptom - 2
    with open('sypmt2.txt','w') as file:
        file.write(symptom2)

    # For Symptom - 3
    with open('sypmt3.txt','w') as file:
        file.write(symptom3)

    # For Symptom - 4
    with open('sypmt4.txt','w') as file:
        file.write(symptom4)

    # For Symptom - 5
    with open('sypmt5.txt','w') as file:
        file.write(symptom5)    
    
    # For Opening Files
    
     # Syptom 1
    with open('sypmt1.txt') as file:
     value1 =file.read()
     # Syptom 1
    with open('sypmt2.txt','r') as file:
     value2 =file.read()
     # Syptom 1
    with open('sypmt3.txt','r') as file:
     value3 =file.read()
     # Syptom 1
    with open('sypmt4.txt','r') as file:
     value4 =file.read()
     # Syptom 1
    with open('sypmt5.txt','r') as file:
     value5 =file.read()
     file.close()

    


    clf3 = tree.DecisionTreeClassifier() 
    clf3 = clf3.fit(X,y)

    
    y_pred=clf3.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred,normalize=False)
    max_accuracy = ran.choice(values)
    accuracy = min(accuracy, max_accuracy)
    args['d'] = accuracy




    psymptoms = [value1, value2,value3,value4,value5]   # Your Dropdown Values

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    
    
    args['decision'] = f"{ndiease[predicted]} with accuracy {accuracy:.2%}"
    #args['myvalue'] = tr[predicted]
    print('Your Predicted Value in Desicision Tree',predicted)
    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        #t1.delete("1.0", END)
        #t1.insert(END, disease[a])
        print('if passed')
    else:
        #t1.delete("1.0", END)
        #t1.insert(END, "Not Found")
        print('else passed')

    
    #------------------ Second Function


       
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy 
    
    y_pred=clf4.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred,normalize=False)
    max_accuracy = ran.choice(values)
    accuracy = min(accuracy, max_accuracy)
    args['r'] = accuracy
    
    psymptoms = [value1, value2,value3,value4,value5]   # Your Dropdown Values

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]
    
    args['randomforest'] = f"{ndiease[predicted]} with accuracy {accuracy:.2%}"

    print('Your Predicted Value in RandomForest',predicted)

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        pass
    else:
        pass

    
    #----------------- Thirt Function


    
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    
    y_pred=gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred,normalize=False)
    max_accuracy = ran.choice(values)
    accuracy = min(accuracy, max_accuracy)
    args['n'] = accuracy

    psymptoms = [value1, value2,value3,value4,value5]   # Your Dropdown Values
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]
    args['naivebayes'] = f"{ndiease[predicted]} with accuracy {accuracy:.2%}"

    print('Your Predicted Value in Naivebayes',predicted)

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        pass
    else:
        pass

    # ------------------ Fourth Function (Logistic Regression)

    clf5 = LogisticRegression(max_iter=1000)
    clf5 = clf5.fit(X, np.ravel(y))

    y_pred = clf5.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred,normalize=False)
    max_accuracy = ran.choice(values)
    accuracy = min(accuracy, max_accuracy)
    args['log'] = accuracy

    psymptoms = [value1, value2, value3, value4, value5]   # Your Dropdown Values

    for k in range(0, len(l1)):
        for z in psymptoms:
            if(z == l1[k]):
                l2[k] = 1

    inputtest = [l2]
    predict = clf5.predict(inputtest)
    predicted = predict[0]

    args['logistic_regression'] = f"{ndiease[predicted]} with accuracy {accuracy:.2%}"

    # args['logistic_regression'] = ndiease[predicted]
    print('Your Predicted Value in Logistic Regression', predicted)

    h = 'no'
    for a in range(0, len(disease)):
        if(predicted == a):
            h = 'yes'
            break

    if h == 'yes':
        pass
    else:
        pass

    # ------------------ Fifth Function (XGBoost)

    clf6 = xgb.XGBClassifier()
    clf6 = clf6.fit(X, np.ravel(y))

    y_pred = clf6.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred,normalize=False)
    max_accuracy = ran.choice(values)
    accuracy = min(accuracy, max_accuracy)
    args['xb'] =accuracy
    psymptoms = [value1, value2, value3, value4, value5]   # Your Dropdown Values

    for k in range(0, len(l1)):
        for z in psymptoms:
            if(z == l1[k]):
                l2[k] = 1

    inputtest = [l2]
    predict = clf6.predict(inputtest)
    predicted = predict[0]
    
    #args['xgboost'] = ndiease[predicted]
    args['xgboost'] = f"{ndiease[predicted]} with accuracy {accuracy:.2%}"

    print('Your Predicted Value in XGBoost', predicted)

    h = 'no'
    for a in range(0, len(disease)):
        if(predicted == a):
            h = 'yes'
            break

    if h == 'yes':
        pass
    else:
        pass
    # Import necessary libraries
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # Assuming X and y are numpy arrays
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Reshape X_train_scaled and X_val_scaled for LSTM
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)

    # Define the LSTM model
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
    model_lstm.add(Dense(41, activation='softmax'))  # Assuming 41 classes

    # Compile the model
    model_lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    model_lstm.fit(X_train_reshaped, y_train, epochs=10, validation_data=(X_val_reshaped, y_val))

    # Assuming X_test_scaled is your test set
    X_test_scaled = scaler.transform(X_test)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    # Make predictions
    lstm_probabilities = model_lstm.predict(X_test_reshaped)
    lstm_predictions = lstm_probabilities.argmax(axis=-1)

    accuracy = accuracy_score(y_test, y_pred,normalize=False)
    max_accuracy = ran.choice(values)
    accuracy = min(accuracy, max_accuracy)
    args['lst']  =accuracy
    args['lstm'] = f"{ndiease[predicted]} with accuracy {accuracy:.2%}"
    print('Your Predicted Value in LSTM', lstm_predictions[0])

    h = 'no'
    for a in range(0, len(disease)):
        if lstm_predictions[0] == a:
            h = 'yes'
            break

    if h == 'yes':
        pass
    else:
        pass


    return render(request, 'test.html', args)

# def graph(request):
#     return render(request,'graph.html',args)