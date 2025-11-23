from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpResponse, JsonResponse
from django.views.decorators.http import require_POST
from django.db import models
from .models import CustomUser
import re

# Create your views here.

def index(request):
    return render(request, 'index.html')

def how_it_works(request):
    return render(request, 'how_it_works.html')

@login_required
def predict(request):
    return render(request, 'predict.html')

def about(request):
    return render(request, 'about.html')

def login_view(request):
    if request.user.is_authenticated:
        return redirect('index')
    
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        if not email or not password:
            messages.error(request, 'Please enter both email and password.')
            return render(request, 'login.html')
        
        # Authenticate user
        user = authenticate(request, email=email, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, f'Welcome back, {user.get_full_name()}!')
            return redirect('index')
        else:
            messages.error(request, 'Invalid email or password. Please try again.')
    
    return render(request, 'login.html')

def register_view(request):
    if request.user.is_authenticated:
        return redirect('index')
    
    if request.method == 'POST':
        full_name = request.POST.get('full_name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        
        # Validation
        if not full_name or not email or not password or not confirm_password:
            messages.error(request, 'Please fill in all fields.')
            return render(request, 'register.html')
        
        if password != confirm_password:
            messages.error(request, 'Passwords do not match.')
            return render(request, 'register.html')
        
        if len(password) < 8:
            messages.error(request, 'Password must be at least 8 characters long.')
            return render(request, 'register.html')
        
        # Check if email already exists
        if CustomUser.objects.filter(email=email).exists():
            messages.error(request, 'An account with this email already exists.')
            return render(request, 'register.html')
        
        # Check if email format is valid
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            messages.error(request, 'Please enter a valid email address.')
            return render(request, 'register.html')
        
        try:
            # Create user
            user = CustomUser.objects.create_user(
                email=email,
                username=email.split('@')[0],  # Use email username as default
                full_name=full_name,
                password=password
            )
            
            messages.success(request, 'Account created successfully! You can now log in.')
            return redirect('login')
            
        except Exception as e:
            messages.error(request, f'An error occurred while creating your account: {str(e)}')
            return render(request, 'register.html')
    
    return render(request, 'register.html')

@login_required
def logout_view(request):
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('login')

@login_required
def dashboard(request):
    return render(request, 'dashboard.html')

# Admin Dashboard Views
def admin_required(user):
    return user.is_authenticated and user.is_staff_member()

@user_passes_test(admin_required, login_url='/login/')
def admin_dashboard(request):
    users = CustomUser.objects.all().order_by('-created_at')
    
    # Filter users based on query parameters
    role_filter = request.GET.get('role', '')
    status_filter = request.GET.get('status', '')
    search_query = request.GET.get('search', '')
    
    if role_filter:
        users = users.filter(role=role_filter)
    
    if status_filter == 'active':
        users = users.filter(is_active=True)
    elif status_filter == 'inactive':
        users = users.filter(is_active=False)
    
    if search_query:
        users = users.filter(
            models.Q(username__icontains=search_query) |
            models.Q(email__icontains=search_query) |
            models.Q(full_name__icontains=search_query)
        )
    
    context = {
        'users': users,
        'total_users': users.count(),
        'active_users': users.filter(is_active=True).count(),
        'inactive_users': users.filter(is_active=False).count(),
        'admin_users': users.filter(role='admin').count(),
        'staff_users': users.filter(role='staff').count(),
        'regular_users': users.filter(role='user').count(),
    }
    
    return render(request, 'admin/admin_dashboard.html', context)

@user_passes_test(admin_required, login_url='/login/')
def admin_user_detail(request, user_id):
    user = get_object_or_404(CustomUser, id=user_id)
    return render(request, 'admin/user_detail.html', {'user': user})

@user_passes_test(admin_required, login_url='/login/')
@require_POST
def admin_toggle_user_status(request, user_id):
    user = get_object_or_404(CustomUser, id=user_id)
    
    # Prevent deactivating yourself
    if user == request.user:
        return JsonResponse({
            'success': False,
            'message': 'You cannot deactivate your own account.'
        })
    
    user.is_active = not user.is_active
    user.save()
    
    status = 'activated' if user.is_active else 'deactivated'
    return JsonResponse({
        'success': True,
        'message': f'User {user.username} has been {status}.',
        'is_active': user.is_active
    })

@user_passes_test(admin_required, login_url='/login/')
@require_POST
def admin_change_user_role(request, user_id):
    user = get_object_or_404(CustomUser, id=user_id)
    
    # Prevent changing your own role
    if user == request.user:
        return JsonResponse({
            'success': False,
            'message': 'You cannot change your own role.'
        })
    
    new_role = request.POST.get('role')
    if new_role not in dict(CustomUser.ROLE_CHOICES):
        return JsonResponse({
            'success': False,
            'message': 'Invalid role selected.'
        })
    
    user.role = new_role
    user.save()
    
    return JsonResponse({
        'success': True,
        'message': f'User {user.username} role changed to {user.get_role_display()}.',
        'new_role': new_role
    })

@user_passes_test(admin_required, login_url='/login/')
@require_POST
def admin_delete_user(request, user_id):
    user = get_object_or_404(CustomUser, id=user_id)
    
    # Prevent deleting yourself or other admins
    if user == request.user:
        return JsonResponse({
            'success': False,
            'message': 'You cannot delete your own account.'
        })
    
    if user.role == 'admin' and not request.user.is_superuser:
        return JsonResponse({
            'success': False,
            'message': 'Only superusers can delete admin accounts.'
        })
    
    username = user.username
    user.delete()
    
    return JsonResponse({
        'success': True,
        'message': f'User {username} has been deleted.'
    })