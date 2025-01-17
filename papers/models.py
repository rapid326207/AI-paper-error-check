from django.db import models
from django.utils import timezone

class Paper(models.Model):
    file = models.FileField(upload_to='papers/')
    title = models.CharField(max_length=255)
    processed = models.BooleanField(default=False)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    error_message = models.TextField(null=True, blank=True)
    has_summary = models.BooleanField(default=False)
    has_analysis = models.BooleanField(default=False)
    def __str__(self):
        return self.title

class PaperAnalysis(models.Model):
    paper = models.ForeignKey('Paper', on_delete=models.CASCADE, related_name='analyses')
    analysis_data = models.JSONField()
    analyzed_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-analyzed_at']

    def __str__(self):
        return f"Analysis for {self.paper.title} at {self.analyzed_at}"

class PaperSummary(models.Model):
    paper = models.ForeignKey('Paper', on_delete=models.CASCADE, related_name='summaries')
    summary_data = models.JSONField()
    generated_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-generated_at']

    def __str__(self):
        return f"Summary for {self.paper.title} at {self.generated_at}"
