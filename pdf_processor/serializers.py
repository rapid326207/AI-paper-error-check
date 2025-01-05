from rest_framework import serializers
from .models import PDFDocument

class PDFDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = PDFDocument
        fields = ['id', 'title', 'file', 'uploaded_at']
