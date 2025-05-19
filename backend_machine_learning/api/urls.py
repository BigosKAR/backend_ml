from django.urls import path
from . import views

urlpatterns = [
    path('api/predict', views.PredictPriceView.as_view()),
]