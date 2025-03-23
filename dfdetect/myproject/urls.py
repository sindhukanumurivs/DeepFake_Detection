from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect

def redirect_to_signup(request):
    return redirect('detect')  # Redirect root URL to signup page

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('allauth.urls')),
    path('detect/', include('myapp.urls')),  # Include your app's URL
    path('', redirect_to_signup),  # Redirect root URL to signup
    
]
