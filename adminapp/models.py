from django.db import models

# Create your models here.

class Upload_dataset_model(models.Model):
    user_id = models.AutoField(primary_key = True)
    Dataset = models.FileField(null=True)
    File_size = models.CharField(max_length = 100) 
    Date_Time = models.DateTimeField(auto_now = True)
    
    class Meta:
        db_table = 'upload_dataset'

# dataset
class DATASET(models.Model):
    DS_ID = models.AutoField(primary_key = True)
    current = models.FloatField()
    inst_energy = models.FloatField() 
    pf = models.FloatField()
    house_interval = models.FloatField()
    voltage = models.FloatField()
    freaquency = models.FloatField()
    day = models.FloatField()
    
    class Meta:
        db_table = 'Dataset'




class BiLSTM_CNN(models.Model):
    LSTM_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'BiLSTM_CNN_algo'

class RFR(models.Model):
    RFR_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'RFR_algo'

class XGBR(models.Model):
    XGBR_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'XGBR_algo'

class LSTM(models.Model):
    LSTM_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'LSTM_algo'

class LGBMR(models.Model):
    LGBMR_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'LGBMR_algo'

class ADA_ALGO(models.Model):
    ADA_ALGO_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'ADA_algo'



class KNN_ALGO(models.Model):
    KNN_ALGO_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'KNN_algo'


class DECISSION_ALGO(models.Model):
    DECISSION_ALGO_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'DECISSION_algo'

class  GradientBoosting(models.Model):
    GradientBoosting_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = ' GradientBoosting_algo'
class   RandomForest(models.Model):
    RandomForest_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'RandomForest_algo'

class   Logistic(models.Model):
    Logistic_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length = 100)
    
    class Meta:
        db_table = 'Logistic_algo'