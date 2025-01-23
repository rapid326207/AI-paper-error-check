from rest_framework import serializers
from .models import Paper
from django.conf import settings
import os

class PaperSerializer(serializers.ModelSerializer):
    class Meta:
        model = Paper
        fields = ['id', 'file', 'title', 'processed', 'has_summary', 'has_analysis', 'created_at', 'updated_at']
    def get_file_path(self, obj):
        if obj.file:
            return os.path.join(settings.MEDIA_ROOT, str(obj.file))
        return None