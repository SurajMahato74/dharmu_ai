from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone
import json

class CustomUser(AbstractUser):
    ROLE_CHOICES = [
        ('user', 'User'),
        ('staff', 'Staff'),
        ('admin', 'Admin'),
    ]
    
    email = models.EmailField(unique=True)
    full_name = models.CharField(max_length=255)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='user')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username', 'full_name']
    
    def __str__(self):
        return self.email
    
    def get_full_name(self):
        return self.full_name
    
    def is_staff_member(self):
        return self.role in ['staff', 'admin']

class PredictionHistory(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='predictions', null=True, blank=True)
    
    # Input configuration
    brand = models.CharField(max_length=50, blank=True)
    processor = models.CharField(max_length=100, blank=True)
    ram = models.CharField(max_length=20, blank=True)
    storage = models.CharField(max_length=20, blank=True)
    storage_type = models.CharField(max_length=20, blank=True)
    display_size = models.CharField(max_length=10, blank=True)
    resolution = models.CharField(max_length=20, blank=True)
    gpu = models.CharField(max_length=100, blank=True)
    os = models.CharField(max_length=50, blank=True)
    weight = models.CharField(max_length=10, blank=True)
    warranty = models.CharField(max_length=10, blank=True)
    spec_rating = models.CharField(max_length=10, blank=True)
    
    # Prediction results
    model_used = models.CharField(max_length=50)
    predicted_price = models.IntegerField()
    formatted_price = models.CharField(max_length=20)
    
    # Suggestions (stored as JSON)
    laptop_suggestions = models.TextField()  # JSON string
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.brand} - ₹{self.predicted_price:,}"
    
    def get_suggestions(self):
        try:
            return json.loads(self.laptop_suggestions)
        except:
            return []
    
    def set_suggestions(self, suggestions):
        self.laptop_suggestions = json.dumps(suggestions)
    
    def get_filled_fields(self):
        fields = {}
        for field in ['brand', 'processor', 'ram', 'storage', 'gpu', 'os']:
            value = getattr(self, field)
            if value:
                fields[field] = value
        return fields