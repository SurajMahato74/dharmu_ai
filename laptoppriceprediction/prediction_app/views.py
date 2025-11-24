from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from datetime import datetime
from .models import PredictionHistory

# Load the improved model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'improved_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'improved_scaler.pkl')
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'improved_encoders.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'improved_features.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print("Improved model loaded successfully!")
except Exception as e:
    print(f"Error loading improved model: {e}")
    model = None
    scaler = None
    label_encoders = None
    feature_names = None

# Store prediction history
prediction_history = []

def extract_numeric_features(data):
    """Extract numeric features from text columns"""
    
    # Extract RAM
    def extract_ram(ram_str):
        if pd.isna(ram_str) or ram_str == '':
            return 8  # Default
        ram_str = str(ram_str).upper()
        if 'GB' in ram_str:
            return int(ram_str.replace('GB', '').strip())
        return 8
    
    # Extract Storage
    def extract_storage(storage_str):
        if pd.isna(storage_str) or storage_str == '':
            return 512  # Default
        storage_str = str(storage_str).upper()
        if 'TB' in storage_str:
            return int(float(storage_str.replace('TB', '').strip()) * 1024)
        elif 'GB' in storage_str:
            return int(storage_str.replace('GB', '').strip())
        return 512
    
    # Extract GPU memory
    def extract_gpu_memory(gpu_str):
        if pd.isna(gpu_str) or gpu_str == '':
            return 0
        gpu_str = str(gpu_str).upper()
        if 'GB' in gpu_str and any(x in gpu_str for x in ['NVIDIA', 'AMD', 'RTX', 'GTX']):
            try:
                # Extract number before GB
                parts = gpu_str.split('GB')[0].split()
                for part in reversed(parts):
                    if part.isdigit():
                        return int(part)
            except:
                pass
        return 0
    
    data['ram_gb'] = extract_ram(data.get('ram', '8GB'))
    data['storage_gb'] = extract_storage(data.get('storage', '512GB'))
    data['gpu_memory'] = extract_gpu_memory(data.get('gpu', 'Intel UHD Graphics'))
    
    return data

def create_engineered_features(data):
    """Create engineered features"""
    
    # Gaming laptop indicator
    gaming_keywords = ['gaming', 'rog', 'tuf', 'predator', 'legion', 'omen', 'nitro', 'alienware']
    laptop_name = str(data.get('name', '')).lower()
    data['is_gaming'] = int(any(keyword in laptop_name for keyword in gaming_keywords))
    
    # Premium brand indicator
    premium_brands = ['Apple', 'Dell', 'HP', 'Lenovo', 'Asus']
    data['is_premium_brand'] = int(data.get('brand', '') in premium_brands)
    
    # Performance score
    spec_rating = float(data.get('spec_rating', 65))
    data['performance_score'] = (
        data['ram_gb'] * 0.3 + 
        data['storage_gb'] * 0.0001 + 
        data['gpu_memory'] * 0.4 + 
        spec_rating * 0.3
    )
    
    # RAM to storage ratio
    data['ram_storage_ratio'] = data['ram_gb'] / (data['storage_gb'] / 1000)
    
    # Screen area
    width = float(data.get('resolution_width', 1920))
    height = float(data.get('resolution_height', 1080))
    data['screen_area'] = width * height / 1000000
    
    return data

def prepare_features_for_prediction(data):
    """Prepare features for prediction"""
    
    # Extract and engineer features
    data = extract_numeric_features(data)
    data = create_engineered_features(data)
    
    # Numerical features
    numerical_features = [
        'ram_gb', 'storage_gb', 'gpu_memory', 'spec_rating',
        'display_size', 'resolution_width', 'resolution_height',
        'is_gaming', 'is_premium_brand', 'performance_score',
        'ram_storage_ratio', 'screen_area'
    ]
    
    # Categorical features
    categorical_features = ['brand', 'OS']
    
    # Create feature vector
    feature_vector = []
    
    # Add numerical features
    for feature in numerical_features:
        if feature == 'spec_rating':
            feature_vector.append(float(data.get(feature, 65)))
        elif feature == 'display_size':
            feature_vector.append(float(data.get(feature, 15.6)))
        elif feature == 'resolution_width':
            feature_vector.append(float(data.get(feature, 1920)))
        elif feature == 'resolution_height':
            feature_vector.append(float(data.get(feature, 1080)))
        else:
            feature_vector.append(data[feature])
    
    # Add encoded categorical features
    for feature in categorical_features:
        value = data.get(feature, 'Unknown')
        if feature in label_encoders:
            try:
                encoded_value = label_encoders[feature].transform([value])[0]
            except ValueError:
                # Handle unknown categories
                encoded_value = 0
        else:
            encoded_value = 0
        feature_vector.append(encoded_value)
    
    return np.array(feature_vector).reshape(1, -1)

def predict_laptop_price(laptop_data):
    """Predict laptop price using the improved model"""
    if model is None:
        return None, "Model not loaded"
    
    try:
        # Prepare features
        features = prepare_features_for_prediction(laptop_data)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Apply minimum price threshold
        prediction = max(10000, prediction)
        
        return prediction, None
        
    except Exception as e:
        return None, str(e)

def index(request):
    """Render the main page"""
    return render(request, 'prediction_app/index.html')

@csrf_exempt
@require_http_methods(["POST"])
def predict_price(request):
    """API endpoint to predict laptop price"""
    try:
        # Handle both JSON and FormData
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            # Handle FormData from HTML form
            data = {}
            for key, value in request.POST.items():
                if key != 'csrfmiddlewaretoken':
                    data[key] = value
        
        # Extract laptop specifications with proper defaults
        def safe_float(value, default):
            try:
                return float(value) if value and value != '' else default
            except (ValueError, TypeError):
                return default
        
        def safe_str(value, default):
            return value if value and value != '' else default
        
        # Parse resolution
        resolution = data.get('resolution', '1920x1080')
        if 'x' in resolution:
            width, height = resolution.split('x')
            res_width = int(width)
            res_height = int(height)
        else:
            res_width = 1920
            res_height = 1080
        
        laptop_specs = {
            'brand': safe_str(data.get('brand'), 'HP'),
            'ram': safe_str(data.get('ram'), '8') + 'GB',
            'storage': safe_str(data.get('storage'), '512') + 'GB',
            'gpu': safe_str(data.get('gpu'), 'Intel UHD Graphics'),
            'processor': safe_str(data.get('processor'), 'Intel Core i5'),
            'OS': safe_str(data.get('os'), 'Windows 11 OS'),
            'spec_rating': safe_float(data.get('spec_rating'), 65),
            'display_size': safe_float(data.get('display_size'), 15.6),
            'resolution_width': res_width,
            'resolution_height': res_height,
            'name': safe_str(data.get('name'), 'Laptop')
        }
        
        # Predict price
        predicted_price, error = predict_laptop_price(laptop_specs)
        
        if error:
            return JsonResponse({
                'success': False,
                'error': error
            }, status=400)
        
        # Save to database
        suggestions = [
            {
                'name': f'{laptop_specs["brand"]} Budget Laptop',
                'full_name': f'{laptop_specs["brand"]} {laptop_specs["ram"]} {laptop_specs["storage"]} Laptop',
                'price': predicted_price * 0.9,
                'category': 'Budget Option',
                'specs': f'{laptop_specs["ram"]} RAM, {laptop_specs["storage"]} Storage, {laptop_specs["gpu"]}',
                'details': 'Great value for money with reliable performance'
            }
        ]
        
        prediction_record = PredictionHistory.objects.create(
            brand=laptop_specs['brand'],
            processor=laptop_specs['processor'],
            ram=laptop_specs['ram'],
            storage=laptop_specs['storage'],
            gpu=laptop_specs['gpu'],
            os=laptop_specs['OS'],
            model_used='Improved Gradient Boosting',
            predicted_price=int(predicted_price),
            formatted_price=f'Rs.{predicted_price:,.0f}',
            laptop_suggestions=json.dumps(suggestions)
        )
        
        # Debug info
        ram_gb = extract_numeric_features(laptop_specs)['ram_gb']
        storage_gb = extract_numeric_features(laptop_specs)['storage_gb']
        gpu_memory = extract_numeric_features(laptop_specs)['gpu_memory']
        
        print(f"Features: Brand={laptop_specs['brand']}, RAM={ram_gb}GB, Storage={storage_gb}GB, GPU={gpu_memory}GB")
        print(f"Prediction: Rs.{predicted_price:,.0f} using improved model")
        
        return JsonResponse({
            'success': True,
            'predicted_price': round(predicted_price, 2),
            'formatted_price': f'₹{predicted_price:,.0f}',
            'currency': 'INR',
            'model_used': 'Improved Gradient Boosting',
            'user_config': {
                'brand': laptop_specs['brand'],
                'processor': laptop_specs['processor'],
                'ram': laptop_specs['ram'],
                'storage': laptop_specs['storage'],
                'gpu': laptop_specs['gpu'],
                'os': laptop_specs['OS']
            },
            'laptop_suggestions': [
                {
                    'name': f'{laptop_specs["brand"]} Budget Laptop',
                    'full_name': f'{laptop_specs["brand"]} {laptop_specs["ram"]} {laptop_specs["storage"]} Laptop',
                    'price': predicted_price * 0.9,
                    'category': 'Budget Option',
                    'specs': f'{laptop_specs["ram"]} RAM, {laptop_specs["storage"]} Storage, {laptop_specs["gpu"]}',
                    'details': 'Great value for money with reliable performance'
                },
                {
                    'name': f'{laptop_specs["brand"]} Recommended',
                    'full_name': f'{laptop_specs["brand"]} Recommended Configuration',
                    'price': predicted_price,
                    'category': 'Best Match',
                    'specs': f'{laptop_specs["ram"]} RAM, {laptop_specs["storage"]} Storage, {laptop_specs["processor"]}',
                    'details': 'Perfect balance of performance and price'
                },
                {
                    'name': f'{laptop_specs["brand"]} Premium',
                    'full_name': f'{laptop_specs["brand"]} Premium Configuration',
                    'price': predicted_price * 1.2,
                    'category': 'Premium Option',
                    'specs': f'Upgraded specs with better performance',
                    'details': 'Enhanced features for demanding users'
                }
            ],
            'features_used': {
                'ram_gb': ram_gb,
                'storage_gb': storage_gb,
                'gpu_memory': gpu_memory,
                'brand': laptop_specs['brand']
            }
        })
        
    except (json.JSONDecodeError, KeyError) as e:
        return JsonResponse({
            'success': False,
            'error': f'Invalid data format: {str(e)}'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@require_http_methods(["GET"])
def prediction_history_view(request):
    """API endpoint to get prediction history"""
    predictions = PredictionHistory.objects.all()[:20]
    history = []
    
    for pred in predictions:
        history.append({
            'timestamp': pred.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'config': pred.get_filled_fields(),
            'model_used': pred.model_used,
            'formatted_price': pred.formatted_price,
            'suggestions': pred.get_suggestions()
        })
    
    return JsonResponse({
        'success': True,
        'history': history,
        'total_predictions': PredictionHistory.objects.count()
    })

@require_http_methods(["GET"])
def model_info(request):
    """API endpoint to get model information"""
    return JsonResponse({
        'success': True,
        'model_type': 'Improved Gradient Boosting Regressor',
        'features': feature_names if feature_names else [],
        'model_loaded': model is not None,
        'description': 'Enhanced model with outlier removal and better feature engineering'
    })

# Additional view functions required by URLs
def how_it_works(request):
    """How it works page"""
    return render(request, 'prediction_app/how_it_works.html')

def predict(request):
    """Predict page"""
    return render(request, 'prediction_app/predict.html')

def about(request):
    """About page"""
    return render(request, 'prediction_app/about.html')

def login_view(request):
    """Login page"""
    return render(request, 'prediction_app/login.html')

def register_view(request):
    """Register page"""
    return render(request, 'prediction_app/register.html')

def logout_view(request):
    """Logout view"""
    from django.contrib.auth import logout
    from django.shortcuts import redirect
    logout(request)
    return redirect('index')

def dashboard(request):
    """Dashboard page"""
    return render(request, 'prediction_app/dashboard.html')

def get_prediction_history(request):
    """Get prediction history - alias for prediction_history_view"""
    return prediction_history_view(request)