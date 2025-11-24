from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser, PredictionHistory

@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    model = CustomUser
    list_display = ('email', 'full_name', 'username', 'role', 'is_staff', 'is_active', 'created_at')
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'role', 'created_at')
    search_fields = ('email', 'full_name', 'username')
    readonly_fields = ('created_at', 'updated_at', 'date_joined', 'last_login')
    
    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        ('Personal info', {'fields': ('first_name', 'last_name', 'email', 'full_name')}),
        ('Permissions', {
            'fields': ('is_active', 'is_staff', 'is_superuser', 'role', 'groups', 'user_permissions'),
        }),
        ('Important dates', {'fields': ('last_login', 'date_joined', 'created_at', 'updated_at')}),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'email', 'full_name', 'password1', 'password2'),
        }),
    )

@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ('brand', 'processor', 'ram', 'storage', 'predicted_price', 'created_at')
    list_filter = ('brand', 'created_at')
    search_fields = ('brand', 'processor')
    readonly_fields = ('created_at',)