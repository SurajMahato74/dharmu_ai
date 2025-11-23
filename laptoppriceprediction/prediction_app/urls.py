from django.urls import path, include
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('how-it-work/', views.how_it_works, name='how_it_work'),
    path('predict/', views.predict, name='predict'),
    path('about/', views.about, name='about'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    
    # Password Reset URLs
    path('password-reset/', auth_views.PasswordResetView.as_view(template_name='registration/password_reset_form.html'), name='password_reset'),
    path('password-reset/done/', auth_views.PasswordResetDoneView.as_view(template_name='registration/password_reset_done.html'), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='registration/password_reset_confirm.html'), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name='registration/password_reset_complete.html'), name='password_reset_complete'),
    
    # Admin Dashboard URLs
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin-dashboard/admin/user/<int:user_id>/', views.admin_user_detail, name='admin_user_detail'),
    path('admin-dashboard/admin/toggle-user-status/<int:user_id>/', views.admin_toggle_user_status, name='admin_toggle_user_status'),
    path('admin-dashboard/admin/change-user-role/<int:user_id>/', views.admin_change_user_role, name='admin_change_user_role'),
    path('admin-dashboard/admin/delete-user/<int:user_id>/', views.admin_delete_user, name='admin_delete_user'),
]