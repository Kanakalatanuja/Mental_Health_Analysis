from django.shortcuts import render,redirect
import urllib.parse   #for send SMS
import urllib.request #for send SMS
import time
import random
from userapp.models import * # * will import everything
from django.conf import settings
from django.core.files.storage import default_storage
from django.conf import settings
from django.contrib import messages
from django.core.mail import send_mail
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

from django.contrib import messages
import pickle
from adminapp.models import *


# Create your views here.
def sendSMS(user, otp, mobile):
    data = urllib.parse.urlencode({
        'username': 'Codebook',
        'apikey': '56dbbdc9cea86b276f6c',
        'mobile': mobile,
        'message': f'Hello {user}, your OTP for account activation is {otp}. This message is generated from https://www.codebook.in server. Thank you',
        'senderid': 'CODEBK'
    })
    data = data.encode('utf-8')
    # Disable SSL certificate verification
    # context = ssl._create_unverified_context()
    request = urllib.request.Request("https://smslogin.co/v3/api.php?")
    f = urllib.request.urlopen(request, data)
    return f.read()



def user_profile(request):
    views_id = request.session['user_id']
    user = UserDetails.objects.get(user_id = views_id)
    if request.method =='POST':
        username = request.POST.get('full_name')
        email = request.POST.get('email_address')
        phone = request.POST.get('Phone_number')
        password = request.POST.get('pass')
        date = request.POST.get('date')
        print(username, email, phone, password,  'data')

        user.user_username = username
        user.user_email = email
        user.user_contact = phone
        user.user_password = password

        if len(request.FILES)!= 0:
            image = request.FILES['image']
            user.user_image = image
            user.user_username = username
            user.user_email = email
            user.user_contact = phone
            user.user_password = password
            user.save()
            messages.success(request, 'Updated Successfully...!')

        else:
            user.user_username = username
            user.user_email = email
            user.user_contact = phone
            user.user_password = password
            user.save()
            messages.success(request, 'Updated Successfully...!')

    return render(request,'user/profile.html', {'i':user})

def user_about(request):
    return render(request,'user/about.html')

def user_admin(request):
    admin_email= "admin@gmail.com"
    admin_password="admin"
    if request.method=="POST":
        admin_e=request.POST.get("admin_email")
        admin_p=request.POST.get("admin_password")
        if (admin_e==admin_email and admin_p==admin_password):
            messages.success(request,'login successfull')
            return redirect("admin_dashboard")
        else:
            messages.error(request,"login credentials was incorrect....")
            return redirect("user_admin")
    
    return render(request,'user/admin.html')
   

def user_contact(request):
    return render(request,'user/contact.html')

def user_dashboard(request):
    prediction_count =  UserDetails.objects.all().count()
    user_id = request.session["user_id"]
    user = UserDetails.objects.get(user_id = user_id)
    return render(request,'user/user-dashboard.html', {'predictions' : prediction_count, 'la' : user})

def user_feedback(request):
    views_id = request.session['user_id']
    user = UserDetails.objects.get(user_id = views_id)
    if request.method == 'POST':
        u_feedback = request.POST.get('feedback')
        u_rating = request.POST.get('rating')
        if not user_feedback:
            return redirect('')
        sid=SentimentIntensityAnalyzer()
        score=sid.polarity_scores(u_feedback)
        sentiment=None
        if score['compound']>0 and score['compound']<=0.5:
            sentiment='positive'
        elif score['compound']>=0.5:
            sentiment='very positive'
        elif score['compound']<-0.5:
            sentiment='very negative'
        elif score['compound']<0 and score['compound']>=-0.5:
            sentiment='negative'
        else :
            sentiment='neutral'
        messages.success(request,'Feedback sent successfully')

        print(sentiment)
        user.star_feedback=u_feedback
        user.star_rating = u_rating
        user.save()
        UserFeedbackModels.objects.create(user_details = user, star_feedback = u_feedback, star_rating = u_rating, sentment= sentiment)
        rev=UserFeedbackModels.objects.filter()
    
    return render(request,'user/user-feedback.html')

def user_forgotpassword(request):
    return render(request,'user/user-forgotpwd.html')

def user_index(request):
    return render(request,'user/user-index.html')


from django.contrib.auth.hashers import check_password

def user_login(request):
    if request.method == 'POST':
        email = request.POST.get('user_email')
        password = request.POST.get('user_password')
        
        print(f"Email entered: {email}, Password entered: {password}")  # Debugging logs

        try:
            # Fetch the user based on email
            user = UserDetails.objects.get(user_email=email)
            print(f"User found: {user}")  # Debugging log to check if the user is fetched
            
            # Check if password matches using check_password
            if check_password(password, user.user_password):
                print(f"Password match for user: {user.user_username}")  # Debugging log for password match
                
                # Check if the user is approved
                if user.user_status == 'Accepted':
                    request.session['user_id'] = user.user_id
                    print('Login successful and session created.')  # Debugging log for successful login

                    messages.success(request, 'Login successful')
                    return redirect('user_dashboard')
                else:
                    messages.info(request, "Your account is still pending approval.")
                    return redirect('user_login')

            else:
                print(f"Password does not match for user: {user.user_username}")  # Debugging log for password mismatch
                messages.error(request, 'Invalid password')
                return redirect('user_login')

        except UserDetails.DoesNotExist:
            print(f"No user found with email: {email}")  # Debugging log if the user is not found
            messages.error(request, 'Invalid email or password')
            return redirect('user_login')
        # except Exception as e:
        #     # Log the exact error for debugging purposes
        #     print(f"Unexpected error: {str(e)}")
        #     messages.error(request, 'An error occurred during login. Please try again.')

    return render(request, 'user/user-login.html')

def user_prediction(request):
    if request.method == 'POST':
        Physical_Health_Interview = request.POST.get('Physical_Health_Interview')
        Care_Options = request.POST.get('Care_Options')
        Medical_Leave = request.POST.get('Medical_Leave')
        Benefits = request.POST.get('Benefits')
        Employee_Count_Company = request.POST.get('Employee_Count_Company')
        Family_History = request.POST.get('Family_History')
        Age = request.POST.get('Age')
        Work_Interfere = request.POST.get('Work_Interfere')
        print(Physical_Health_Interview,Care_Options,Medical_Leave,Benefits,Employee_Count_Company,Family_History,Age,Work_Interfere,'data')
        Physical_Health_Interview = int(Physical_Health_Interview)
        Care_Options = int(Care_Options)
        Medical_Leave = int(Medical_Leave)
        Benefits = int(Benefits)
        Family_History = int(Family_History)
        Age = int(Age)
        Employee_Count_Company = int(Employee_Count_Company)
        Work_Interfere = int(Work_Interfere)
        file_path = 'bilstm_cnn.pkl' 

        with open(file_path,'rb') as file:
            loaded_model = pickle.load(file)
       

        res = loaded_model.predict([[ Age,Physical_Health_Interview,Care_Options,Medical_Leave,Benefits,Employee_Count_Company,Family_History,Work_Interfere]])


        dataset = Upload_dataset_model.objects.last()
        df=pd.read_csv('MentalHealth_dataset\health_clean.csv')
        X = df.drop('Treatment', axis = 1)
        y = df['Treatment']
        res_list = res.tolist()
        request.session['result'] = res_list
        print(res_list,'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
      

        return redirect('user_result')
    return render(request,'user/prediction.html')

# from django.shortcuts import render
from adminapp.models import BiLSTM_CNN  # Import your model
def user_result(request):
    model_stats = BiLSTM_CNN.objects.last()
    context = {
        "accuracy": model_stats.Accuracy,
        "recall": model_stats.Recall,
        "Precession": model_stats.Precession,  # Keep as 'Precession' if model can't be changed
        "f1": model_stats.F1_Score
    }
    return render(request, 'user/result.html', context)

# def user_result(request):
#     # Fetch the latest BiLSTM + CNN model statistics
#     model_stats = BiLSTM_CNN.objects.last()

#     # Handle case when there is no data
#     if not model_stats:
#         context = {
#             "accuracy": "N/A",
#             "recall": "N/A",
#             "precision": "N/A",
#             "f1": "N/A",
#             "message": "No results available yet."
#         }
#     else:
#         context = {
#             "accuracy": model_stats.Accuracy,
#             "recall": model_stats.Recall,
#             "precision": model_stats.Precession,  
#             "f1": model_stats.F1_Score,
#             "message": "Model results successfully loaded."
#         }

#     return render(request, 'user/result.html',context)

# def user_result(request):
    
#    return render(request,'user/result.html')

# def user_result(request):
#     messages.success(request,"Social Crime Has Been Detected")
#     accuracy = request.session.get('acr')
#     precession = request.session.get('pre')
#     recall = request.session.get('rec')
#     f1 = request.session.get('f')
#     res = request.session.get('result')
#     print(accuracy,"ttttttttttttttttttttttttt")

#     print(res,"xaxfdgdyfur")

#     return render(request,'user/result.html',{'accuracy':accuracy, 'precession':precession, 'recall': recall, 'f1':f1, 'result':res})

# def user_otp(request):
#     user_id=request.session["user_email"]
#     user=UserDetails.objects.get(user_email=user_id)
#     print(user_id)
#     print(user,"user")
#     print(user.otp,"create_otp")
#     if request.method=="POST":
#         u_otp=request.POST.get("user_otp")
#         u_otp=int(u_otp)
#         print(u_otp,"otp")
#         if u_otp==user.otp:
#             print("if")
#             user.otp_status="verified"
#             user.save()
#             return redirect ("user_login")
#         else:
#             print("else")
#             return redirect("user_otp")

#     return render(request,'user/otp.html')


from django.contrib.auth.hashers import make_password

def user_register(req):
    if req.method == "POST":
        username = req.POST.get("register_username")
        email = req.POST.get("register_email")
        password = req.POST.get("register_password")
        contact = req.POST.get("register_contact")
        profile = req.FILES["register_choosefile"]
        
        # Hash the password before saving it to the database
        hashed_password = make_password(password)
        
        print(username, email, password, contact, profile, "register")
        try:
            user_data = UserDetails.objects.get(user_email=email)
            return redirect("user_register")
        except UserDetails.DoesNotExist:
            UserDetails.objects.create(
                user_username=username,
                user_email=email,
                user_password=hashed_password,  # Store hashed password
                user_contact=contact,
                user_image=profile
            )
            req.session["user_email"] = email
            return redirect("user_login")
  
    return render(req, 'user/register.html')

def userlogout(request):
    view_id = request.session["user_id"]
    user = UserDetails.objects.get(user_id = view_id)
    t = time.localtime()
    user.last_login_time = t
    current_time = time.strftime('%H:%M:%S', t)
    user.last_login_time = current_time
    current_date = time.strftime('%Y-%m-%d')
    user.last_login_date = current_date
    user.save()
    messages.info(request, 'You are logged out..')
    # print(user.Last_Login_Time)
    # print(user.Last_Login_Date)
    return redirect('user_login')



