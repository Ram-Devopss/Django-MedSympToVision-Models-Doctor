from django.shortcuts import render
from django.http import HttpResponse

def aboutus(request):
    return render(request,'aboutus.html')