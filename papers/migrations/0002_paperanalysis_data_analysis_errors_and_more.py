# Generated by Django 5.0.2 on 2025-01-21 04:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('papers', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='paperanalysis',
            name='data_analysis_errors',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='paperanalysis',
            name='logical_framework_errors',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='paperanalysis',
            name='math_errors',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='paperanalysis',
            name='methdology_errors',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='paperanalysis',
            name='research_quality_errors',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='paperanalysis',
            name='technical_presentation_errors',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='paperanalysis',
            name='total_errors',
            field=models.IntegerField(default=0),
        ),
    ]
