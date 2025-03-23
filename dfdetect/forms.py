from django import forms
from .models import UploadedVideo
from .models import ImageUpload

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedVideo
        fields = ['video_file']
class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = ImageUpload
        fields = ["image_file", "image_folder"]
