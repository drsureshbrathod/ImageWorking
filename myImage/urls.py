from django.contrib import admin
from django.urls import path
from . import  views

urlpatterns = [
    path('', views.hotel_image_view,name='index'),
    path('success', views.success, name='success'),
    path('disp',views.display_hotel_images,name='display'),
    path('rem',views.backRemoval,name='remove'),
    path('trans',views.dotransperarnt,name='display'),
    path('newima/',views.newTryImage,name="newImage"),
]