from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request, 'index.html')

def how_it_works(request):
    return render(request, 'how_it_works.html')

def predict(request):
    return render(request, 'predict.html')


def about(request):
    return render(request, 'about.html')


def login_view(request):
    return render(request, 'login.html')

def register_view(request):
    return render(request, 'register.html')