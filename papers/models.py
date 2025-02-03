from django.db import models
from django.utils import timezone
from storages.backends.s3 import S3File
from storages.backends.s3boto3 import S3Boto3Storage
from uuid import uuid4
from django.conf import settings    
import boto3

class PaperFileS3Storage(S3Boto3Storage):
    location = "papers"  # This will be the first suffix in a S3 path <s3>/papers

class SpeechFileS3Storage(S3Boto3Storage):
    location = "speeches"  # This will be the first suffix in a S3 path <s3>/speeches

def get_paper_s3_file_path(instance: "Paper", filename: str):
    return f"{filename}"

def get_speech_s3_file_path(instance: "PaperSpeech", filename: str):
    return f"{uuid4().hex}.{filename.split(".")[-1]}"

class Paper(models.Model):
    file = models.FileField(
        max_length=255,
        storage=PaperFileS3Storage,
        upload_to=get_paper_s3_file_path,
    )
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
    def open(self) -> S3File:
        storage = PaperFileS3Storage()
        return storage.open(self.file.name, mode="rb")

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

class BaseSpeechModel(models.Model):
    SPEECH_TYPE_PAPER_TITLE = 'paper_title'
    SPEECH_TYPE_CHILD_SUMMARY = 'child_summary'
    SPEECH_TYPE_COLLEGE_SUMMARY = 'college_summary'
    SPEECH_TYPE_PHD_SUMMARY = 'phd_summary'
    SPEECH_TYPE_CHOICES = [
        (SPEECH_TYPE_PAPER_TITLE, 'Paper Title'),
        (SPEECH_TYPE_CHILD_SUMMARY, 'Child Summary'),
        (SPEECH_TYPE_COLLEGE_SUMMARY, 'College Summary'),
        (SPEECH_TYPE_PHD_SUMMARY, 'Phd Summary'),
    ]
    
    VOICE_TYPE_CHOICES = [
        ('alloy', 'Alloy'),
        ('ash', 'Ash'),
        ('coral', 'Coral'),
        ('echo', 'Echo'),
        ('fable', 'Fable'),
        ('onyx', 'Onyx'),
        ('nova', 'Nova'),
        ('sage', 'Sage'),
        ('shimmer', 'Shimmer'),
    ]

    speech_type = models.CharField(max_length=20, choices=SPEECH_TYPE_CHOICES)
    voice_type = models.CharField(max_length=20, choices=VOICE_TYPE_CHOICES, default='alloy')
    audio_file = models.FileField(
        storage=SpeechFileS3Storage,
        null=True,
        blank=True,
        upload_to=get_speech_s3_file_path,
    )
    generated_at = models.DateTimeField(auto_now_add=True)
    input_tokens = models.BigIntegerField(default=0)
    output_tokens = models.BigIntegerField(default=0)
    generation_cost = models.FloatField(default=0)
    status = models.CharField(max_length=20, default='pending', choices=[
        ('pending', 'Pending'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ])
    error_message = models.TextField(null=True, blank=True)

    class Meta:
        abstract = True

class PaperSpeech(BaseSpeechModel):
    paper = models.ForeignKey(
        'Paper',
        on_delete=models.CASCADE,
        related_name='speeches'
    )
    content_source = models.TextField(null=True, blank=True)
    allowed_user_ids = models.JSONField(
        default=list,
        blank=True,
        help_text="List of user IDs that have purchased access to this speech"
    )
    payment_records = models.JSONField(
        default=list,
        blank=True,
        help_text="Stores payment metadata in format [{'user_id': '...', 'payment_ref': '...', 'date': '...'}]"
    )

    def get_audio_url(self):
        """Return URL for audio file"""
        if self.audio_file:
            s3 = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_S3_REGION_NAME
            )
            presigned_url = s3.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': settings.AWS_STORAGE_BUCKET_NAME,
                    'Key': self.audio_file.name
                },
                ExpiresIn=604800  # 7 days
            )
            return presigned_url
        else:
            raise ValueError("Audio file does not exist.")

    def generate_speech(self):
        """Generate speech using OpenAI and store to S3"""
        from .services import generate_speech_task  # Celery task
        
        try:
            # Trigger async task
            self.status = 'pending'
            self.save()
            return generate_speech_task(self.id)
        except Exception as e:
            self.status = 'failed'
            self.error_message = str(e)
            self.save()

    @classmethod
    def create_for_source(cls, paper, source, speech_type, voice_type='alloy'):
        """Factory method to create speech entries"""
        speech = cls.objects.create(
            paper=paper,
            speech_type=speech_type,
            voice_type=voice_type,
            status='pending'
        )
        speech.content_source = source 
        speech.save()
        data = speech.generate_speech()
        print(data)
        return data

    def add_user_access(self, user_id: str, payment_reference: str):
        """Add user to allowed list and record payment"""
        if user_id not in self.allowed_user_ids:
            self.allowed_user_ids.append(user_id)
            
        self.payment_records.append({
            'user_id': user_id,
            'payment_ref': payment_reference,
            'date': timezone.now().isoformat()
        })
        self.save()

    def remove_user_access(self, user_id: str):
        """Remove user from allowed list"""
        if user_id in self.allowed_user_ids:
            self.allowed_user_ids.remove(user_id)
            self.save()

    def has_access(self, user_id: str) -> bool:
        """Check if user has access"""
        return user_id in self.allowed_user_ids

    class Meta:
        indexes = [
            models.Index(fields=['allowed_user_ids'], name='user_access_idx')
        ]