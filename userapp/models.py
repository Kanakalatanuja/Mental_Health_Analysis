from django.db import models
class UserDetails(models.Model):
    user_id=models.AutoField(primary_key=True)
    user_email=models.EmailField(max_length=50,null=True)
    user_username=models.TextField(max_length=50,null=True)
    user_contact=models.TextField(max_length=50,null=True)
    user_password=models.TextField(max_length=50,null=True)
    user_image=models.FileField(upload_to="images/",null=True)
    user_status=models.TextField(max_length=50,null=True,default="pending")
    user_feedback=models.TextField(max_length=1000,null=True)
    message=models.TextField(max_length=250,null=True)

    class Meta:
        db_table="UserDetails"


class UserFeedbackModels(models.Model):
    feed_id = models.AutoField(primary_key=True)
    star_feedback = models.TextField(max_length=900)
    star_rating = models.IntegerField()
    star_Date = models.DateTimeField(auto_now_add=True, null=True)
    user_details = models.ForeignKey(UserDetails, on_delete=models.CASCADE)
    sentment = models.TextField(max_length=20,null=True)
    class Meta:
        db_table = 'UserFeedbackModels'



class Dataset(models.Model):
   Data_id = models.AutoField(primary_key=True)
   Image = models.ImageField(upload_to='media/') 
   class Meta:
        db_table = "upload" 

class PredictionCount(models.Model):
    user = models.OneToOneField(UserDetails, on_delete=models.CASCADE)
    prediction_count = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f'{self.user.username} - Predictions: {self.prediction_count}'

class Predict_details(models.Model):
    predict_id = models.AutoField(primary_key=True)
    Field_1 = models.CharField(max_length = 60, null = True)
    Field_2 = models.CharField(max_length = 60, null = True)
    Field_3 = models.CharField(max_length = 60, null = True)
    Field_4 = models.CharField(max_length = 60, null = True)
    Field_5 = models.CharField(max_length = 60, null = True)
    Field_6 = models.CharField(max_length = 60, null = True)
    Field_7 = models.CharField(max_length = 60, null = True)
   
    
    class Meta:
        db_table = "predict_detail"


        


