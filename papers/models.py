from django.db import models
from django.utils import timezone

class Paper(models.Model):
    file = models.FileField(upload_to='papers/', max_length=255)
    title = models.CharField(max_length=500, blank=True)
    processed = models.BooleanField(default=False)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    error_message = models.TextField(null=True, blank=True)
    has_summary = models.BooleanField(default=False)
    has_analysis = models.BooleanField(default=False)
    input_tokens = models.BigIntegerField(default=0)
    output_tokens = models.BigIntegerField(default=0)
    total_cost = models.FloatField(default=0)
    def __str__(self):
        return self.title

class PaperAnalysis(models.Model):
    paper = models.OneToOneField('Paper', on_delete=models.CASCADE, related_name='analysis')  
    analysis_data = models.JSONField()
    analyzed_at = models.DateTimeField(default=timezone.now)
    total_errors = models.IntegerField(default=0)
    math_errors = models.IntegerField(default=0)
    methodology_errors = models.IntegerField(default=0)
    logical_framework_errors = models.IntegerField(default=0)
    data_analysis_errors = models.IntegerField(default=0)
    technical_presentation_errors = models.IntegerField(default=0)
    research_quality_errors = models.IntegerField(default=0)

    class Meta:
        ordering = ['-analyzed_at']

    def __str__(self):
        return f"Analysis for {self.paper.title} at {self.analyzed_at}"

class PaperSummary(models.Model):
    paper = models.OneToOneField('Paper', on_delete=models.CASCADE, related_name='summaries')
    summary_data = models.JSONField()
    generated_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-generated_at']

    def __str__(self):
        return f"Summary for {self.paper.title} at {self.generated_at}"
