from rest_framework import serializers
from .models import Paper

class PaperSerializer(serializers.ModelSerializer):
    class Meta:
        model = Paper
        fields = ['id', 'file', 'title', 'processed', 'has_summary', 'has_analysis', 'created_at', 'updated_at']
