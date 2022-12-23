from django.urls import path
from . import views

urlpatterns = [
    path('',views.home, name='home'),
    path('home',views.home, name='home'),
    path('insuranceForm',views.getInsureFormDetails, name="insuranceForm"),
    path('makePrediction',views.makePrediction, name="makePrediction"),
    path('fileUpload',views.fileUpload,name="fileUpload"),
]
