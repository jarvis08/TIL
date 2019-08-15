from django.urls import path, include
# from artii import views as artii_views
from . import views

urlpatterns = [
    # path('artii/', include('artii.urls')),
    path('', views.artii),
    # path('artii/result/', views.artii_result),
    path('result/', views.artii_result),
]