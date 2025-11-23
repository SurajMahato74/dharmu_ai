from django.urls import path, include
from . import views
urlpatterns = [
    path('', views.index, name='index'),
    path('how-it-work/', views.how_it_works, name='how_it_work'),
    path('predict/', views.predict, name='predict'),
    path('about/', views.about, name='about'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
]