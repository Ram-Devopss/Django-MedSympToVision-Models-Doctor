from django.contrib import admin
from django.urls import path
from homepage import views as v1
from imageprediction import views as v2 
from symptomprediction import views as v3
from aboutus import views as v4
from ximagepredictions import views as v5


urlpatterns = [
    path('',v1.home),
    path('x_ray/',v5.x_ray,name="X Ray Image Predictions"),
    path('image/',v2.predict_it),
    path('aboutus/',v4.aboutus),
    path('symptomprediction/',v3.symptom),
    path('symptomprediction/makeme',v3.makeme,name='makeme'),
    path('home/',v1.home),
]