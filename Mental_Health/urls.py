"""
URL configuration for mental_health_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from adminapp import views as admin_views
from userapp import views as user_views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    #admin views
    path('admin/', admin.site.urls),
    path('admin-dashboard',admin_views.admin_dashboard,name='admin_dashboard'),
    path('admin_pendingusers',admin_views.admin_pendingusers,name='admin_pendingusers'),
    path('all-users',admin_views.all_users,name='all_users'),
    path('accept-user/<int:id>', admin_views.accept_user, name = 'accept_user'),
    path('reject-user/<int:id>', admin_views.reject_user, name = 'reject'),
    path('delete-user/<int:id>', admin_views.delete_user, name = 'delete_user'),
    path('delete-dataset/<int:id>', admin_views.delete_dataset, name = 'delete_dataset'),
    path('adminlogout',admin_views.adminlogout, name='adminlogout'),
    path('upload-dataset',admin_views.upload_dataset,name="upload_dataset"),
    path('view-dataset', admin_views.viewdataset, name = 'viewdataset'),
    path('view-view', admin_views.view_view, name='view_view'),
    path('xgb-algm', admin_views.xgbalgm, name = 'xgbalgm'),
    path('XGBOOST-btn', admin_views.XGBOOST_btn, name='XGBOOST_btn'),
    path('adab-algm', admin_views.adabalgm, name = 'adabalgm'),
    path('ADABoost-btn', admin_views.ADABoost_btn, name='ADABoost_btn'),
    path('knn-algm', admin_views.knnalgm, name = 'knnalgm'),
    path('KNN-btn', admin_views.KNN_btn, name='KNN_btn'),  
    path('logistic', admin_views.logistic, name = 'logistic'),
    path('logistic-btn', admin_views.logistic_btn, name='logistic_btn'),
    path('random', admin_views.random, name = 'random'),
    path('randomforest-btn', admin_views.randomforest_btn, name='randomforest_btn'),
    path('dt-algm', admin_views.dtalgm, name = 'dtalgm'),
    path('Decisiontree-btn', admin_views.Decisiontree_btn, name='Decisiontree_btn'),
    path('GD-alg', admin_views.gdalgm, name = 'gdalgm'),
    path('GD-btn', admin_views.GD_btn, name='GD_btn'),
    path('admin-graph', admin_views.admin_graph, name = 'admin_graph'),
    path('user-feedbacks',admin_views.user_feedbacks,name='user_feedbacks'),
    path('user-sentiment',admin_views.user_sentiment,name='user_sentiment'),
    path('user-graph',admin_views.user_graph,name='user_graph'),
    path("admin-change-status/<int:id>",admin_views.Change_Status,name="change_status"),
    path("delete_user/<int:id>",admin_views.delete_User,name="delete_user"),
    path("adminrejectbtn/<int:x>",admin_views.Admin_Accept_Button,name="adminaccept"),
    path("adminacceptbtn/<int:x>",admin_views.Admin_Reject_Btn,name="adminreject"),
    path('admin_dataset_btn',admin_views.admin_dataset_btn,name='admin_dataset_btn'),
    path('BiLSTM_CNN_btn',admin_views.BiLSTM_CNN_btn, name='BiLSTM_CNN_btn'),
    path('bilstm_cnn', admin_views.bilstm_cnn, name='bilstm_cnn'),




    
    #user views
    path('user-profile',user_views.user_profile,name='user_profile'),
    path('user-about',user_views.user_about,name='user_about'),
    path('user-admin',user_views.user_admin,name='user_admin'),
    path('user-contact',user_views.user_contact,name='user_contact'),
    path('user-dashboard',user_views.user_dashboard,name='user_dashboard'),
    path('user-feedback',user_views.user_feedback,name='user_feedback'),
    path('user-forgotpassword',user_views.user_forgotpassword,name='user_forgotpassword'),
    path('',user_views.user_index,name='user_index'),
    path('user-login',user_views.user_login,name='user_login'),
    path('user-prediction',user_views.user_prediction,name='user_prediction'),
    #path('user-otp',user_views.user_otp,name='user_otp'),
    path('user-register',user_views.user_register,name='user_register'),
    path('userlogout/', user_views.userlogout, name = 'userlogout'),
    path('user-result',user_views.user_result,name='user_result')
]+ static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
