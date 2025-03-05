from django.shortcuts import render,redirect
from django.core.paginator import Paginator
from userapp.models import*
from django.contrib import messages
from django.conf import settings
from django.contrib import messages
from userapp.models import *
from adminapp.models import *
import urllib.request
import urllib.parse
import pandas as pd
from sklearn.ensemble  import AdaBoostClassifier
from sklearn.svm  import SVC
from django.core.paginator import Paginator
from xgboost import XGBClassifier
# Create your views here.
def admin_dashboard(request):
    all_users_count =  UserDetails.objects.all().count()
    pending_users_count = UserDetails.objects.filter(user_status = 'pending').count()
    rejected_users_count = UserDetails.objects.filter(user_status = 'Rejected').count()
    accepted_users_count = UserDetails.objects.filter(user_status = 'Accepted').count()
    datasets_count = UserDetails.objects.all().count()
    no_of_predicts = UserDetails.objects.all().count()
    messages.success(request,"login Successful")
    return render(request,'admin/admin-dashboard.html', {'a' : pending_users_count, 'b' : all_users_count, 'c' : rejected_users_count, 'd' : accepted_users_count, 'e' : datasets_count, 'f' : no_of_predicts})

def admin_dataset_btn(req):
    messages.success(req, 'Dataset Total:6442 files uploaded successfully')
    return redirect('admin_uploaddataset') 

def admin_pendingusers(request):
    users=UserDetails.objects.filter(user_status="pending")
    context={"u":users}
    return render(request,'admin/admin-pendingusers.html',context)

def all_users(request):
    a = UserDetails.objects.all()
    paginator = Paginator(a, 5) 
    page_number = request.GET.get('page')
    post = paginator.get_page(page_number)
    return render(request,'admin/admin-allusers.html',{'all':post})

 
def accept_user(request, id):
    return redirect(request,'admin_pendingusers')

def reject_user(request, id):
    return redirect(request,'admin_pendingusers')

def delete_user(request, id):
    return redirect('all_users')



def delete_dataset(request, id):
    dataset = Upload_dataset_model.objects.get(user_id = id).delete()
    messages.warning(request, 'Dataset was deleted..!')
    return redirect('viewdataset')

def adminlogout(request): 
    return redirect('admin_login')

def upload_dataset(request):
    if request.method == 'POST':
        file = request.FILES['data_file']
        file_size = str((file.size)/1024) +' kb'
        Upload_dataset_model.objects.create(File_size = file_size, Dataset = file)
        messages.success(request, 'Your dataset was uploaded..')
    return render(request,'admin/admin-uploaddataset.html')

def viewdataset(request):
    dataset = Upload_dataset_model.objects.all()
    paginator = Paginator(dataset, 5)
    page_number = request.GET.get('page')
    post = paginator.get_page(page_number)
    return render(request,'admin/admin-viewdataset.html', {'data' : dataset, 'user' : post})

def view_view(request):
    data = Upload_dataset_model.objects.last()
    print(data,type(data),'sssss')
    file = str(data.Dataset)
    df = pd.read_csv(f'./media/{file}')
    table = df.to_html(table_id='data_table')
    return render(request,'admin/admin-view-view.html', {'t':table})

def xgbalgm(request):
    return render(request,'admin/xg-boost.html')

from django.shortcuts import render
from django.contrib import messages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import pickle
from .models import Upload_dataset_model, XGBR  # Adjust based on your actual models


def XGBOOST_btn(request):
    dataset = Upload_dataset_model.objects.last()
    df=pd.read_csv('MentalHealth_dataset\health_clean.csv')

    # Separate features and target
    X = df.drop('Treatment', axis=1)
    y = df['Treatment']

    # Encode categorical columns using LabelEncoder
    label_encoder = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = label_encoder.fit_transform(X[col].astype(str))

    # Scale numerical columns using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert target to binary
    y = label_encoder.fit_transform(y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Convert data to DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set XGBoost parameters for lower accuracy
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.5,          # Higher learning rate (faster, less precise)
        'max_depth': 3,      # Shallower trees
        'subsample': 0.5,    # Lower subsample ratio
        'colsample_bytree': 0.5,  # Lower column sampling ratio
        'random_state': 1
    }

    # Train the model with fewer rounds and early stopping
    evals = [(dtrain, 'train'), (dtest, 'test')]
    bst = xgb.train(params, dtrain, num_boost_round=30, evals=evals, early_stopping_rounds=5, verbose_eval=True)

    # Predict on train and test sets
    train_pred = (bst.predict(dtrain) > 0.5).astype("int")
    test_pred = (bst.predict(dtest) > 0.5).astype("int")

    # Metrics Calculation
    accuracy = round(accuracy_score(y_test, test_pred) * 100, 2)
    precision = round(precision_score(y_test, test_pred) * 100, 2)
    recall = round(recall_score(y_test, test_pred) * 100, 2)
    f1 = round(f1_score(y_test, test_pred) * 100, 2)

    print(f"Accuracy: {accuracy}%")
    print(f"Precision: {precision}%")
    print(f"Recall: {recall}%")
    print(f"F1 Score: {f1}%")

    # Save the model
    model_path = 'xgb_lower_accuracy_model.json'
    bst.save_model(model_path)
    print(f"Model saved as '{model_path}'")

    # Save metrics and model info to the database
    XGBR.objects.create(Accuracy=accuracy, Precession=precision, F1_Score=f1, Recall=recall, Name="XGBoost Algorithm")

    # Cross-validation score
    full_dmatrix = xgb.DMatrix(X, label=y)
    score = cross_val_score(xgb.XGBClassifier(**params), X, y, cv=5)
    print(f"Cross-validation Score: {score.mean()}")

    # Display success message
    messages.success(request, 'XGBoost Algorithm executed successfully.')
    data = XGBR.objects.last()
    return render(request, 'admin/xg-boost.html', {'i': data})
from django.shortcuts import render
from django.contrib import messages
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional, LSTM, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from .models import Upload_dataset_model, BiLSTM_CNN  # Adjust based on your models
def BiLSTM_CNN_btn(request):
    # Load the dataset
    df = pd.read_csv('MentalHealth_dataset\health_clean.csv')

    # Separate features and target
    X = df.drop('Treatment', axis=1)
    y = df['Treatment']

    # Encode categorical columns using LabelEncoder
    label_encoder = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = label_encoder.fit_transform(X[col].astype(str))

    # Scale numerical columns using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert target to binary
    y = label_encoder.fit_transform(y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Reshape input data for LSTM
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Build the BiLSTM+CNN model
    model = Sequential()
    # CNN Layers
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    # First BiLSTM Layer
    model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))
    # Second BiLSTM Layer
    model.add(Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)))
    # Dense Layers
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    # Output Layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)

    # Train the model
    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=32,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping, reduce_lr])

    # Evaluate the model
    train_pred = (model.predict(X_train) > 0.5).astype("int32")
    test_pred = (model.predict(X_test) > 0.5).astype("int32")

    # Metrics Calculation
    accuracy = round(accuracy_score(y_test, test_pred) * 100, 2)
    precision = round(precision_score(y_test, test_pred) * 100, 2)
    recall = round(recall_score(y_test, test_pred) * 100, 2)
    f1 = round(f1_score(y_test, test_pred) * 100, 2)

    # Print evaluation results
    print(f"Accuracy: {accuracy}%")
    print(f"Precision: {precision}%")
    print(f"Recall: {recall}%")
    print(f"F1 Score: {f1}%")

    # Save the model
    model.save('optimized_bilstm_cnn_model.h5')
    print("Model saved as 'optimized_bilstm_cnn_model.h5'")

    # Save metrics and model info to the database
    BiLSTM_CNN.objects.create(Accuracy=accuracy, Precession=precision, F1_Score=f1, Recall=recall, Name="BiLSTM + CNN Algorithm")

    # Cross-validation score (optional)
    # score = cross_val_score(model, X, y, cv=5)
    # print(f"Cross-validation Score: {score.mean()}")

    # Display success message
    messages.success(request, 'BiLSTM + CNN Algorithm executed successfully.')
    data = BiLSTM_CNN.objects.last()
    return render(request, 'admin/bilstm_cnn.html', {'i': data})
def bilstm_cnn(request):
    return render(request,'admin/bilstm_cnn.html')
def adabalgm(request):
    return render(request,'admin/ada-boost.html')

def ADABoost_btn(request):
    dataset = Upload_dataset_model.objects.last()
    # print(dataset.Dataset)
    df=pd.read_csv('MentalHealth_dataset\health_clean.csv')

    X = df.drop('Treatment', axis = 1)
    y = df['Treatment']
    from imblearn.over_sampling import SMOTE

    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)
    # Fit the grid search to the data
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score

    # Create AdaBoostClassifier
    ada = AdaBoostClassifier(random_state=42)

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
        # Add other hyperparameters you want to tune
    }

    # Create GridSearchCV object
    grid_search = GridSearchCV(ada, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Print the best parameters found by GridSearchCV
    print("Best Parameters:", grid_search.best_params_)

    # Get the best model
    best_ada_model = grid_search.best_estimator_

    # Prediction
    train_prediction = best_ada_model.predict(X_train)
    test_prediction = best_ada_model.predict(X_test)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
    recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
    f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)    # Evaluation
    print('*'*20)
    print(accuracy, precession,recall,f1,'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    print('Test accuracy:', accuracy_score(y_test, test_prediction))
    print('Test precision_score:', precision_score(y_test, test_prediction))
    print('Test recall_score:', recall_score(y_test, test_prediction))
    print('Test f1_score:', f1_score(y_test, test_prediction))
    print('Train accuracy:', accuracy_score(y_train, train_prediction))
    print('*'*20)

    # Cross-validation score
    best_score = grid_search.best_score_
    print(f"Best Cross-validation Score: {best_score:.4f}")
    print('*'*20)

    # Prediction Summary by species
    # print(classification_report(y_test, test_prediction))
    print('*'*20)
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "ADA Boost Algorithm"
    ADA_ALGO.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = ADA_ALGO.objects.last()
    messages.success(request, 'Algorithm executed Successfully')
    # Accuracy score
    ada_h = accuracy_score(test_prediction, y_test)
    print(f"{round(ada_h*100, 2)}% Accurate")

    return render(request,'admin/ada-boost.html',{'i':data})

def knnalgm(request):
    return render(request,'admin/knn-algorithem.html')

def KNN_btn(request):
    dataset = Upload_dataset_model.objects.last()
    # print(dataset.Dataset)
    df=pd.read_csv('MentalHealth_dataset\health_clean.csv')

    X = df.drop('Treatment', axis = 1)
    y = df['Treatment']
    from imblearn.over_sampling import SMOTE



    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    # Convert X_train and X_test to NumPy arrays
    X_train_np = np.array(X_train)
    X_test_np = np.array(X_test)

    # Model
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train_np, y_train)

    # Prediction
    train_prediction = knn_model.predict(X_train_np)
    test_prediction = knn_model.predict(X_test_np)

    # Evaluation
    print('*'*20)
    from sklearn.metrics import accuracy_score
    print('Test accuracy:', accuracy_score(y_test, test_prediction))
    print('Train accuracy:', accuracy_score(y_train, train_prediction))

    print('*'*20)
    result = confusion_matrix(y_test, test_prediction)
    print("Confusion Matrix:")
    print(result)

    print('*'*20)
    # Prediction Summary by species
    print(classification_report(y_test, test_prediction))

    print('*'*20)
    # Accuracy score
    Knn_SC = accuracy_score(test_prediction, y_test)
    print(f"{round(Knn_SC*100, 2)}% Accurate")
    # evaluation
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
    recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
    f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "KNN Algorithm"
    KNN_ALGO.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = KNN_ALGO.objects.last()
    messages.success(request, 'Algorithm executed Successfully')


    
    return render(request,'admin/knn-algorithem.html',{'i':data})

def logistic(request):
    return render(request,'admin/Logistic.html')

def logistic_btn(request):
    dataset = Upload_dataset_model.objects.last()
    # print(dataset.Dataset)
    df=pd.read_csv('MentalHealth_dataset\health_clean.csv')
    
    X = df.drop('Treatment', axis = 1)
    y = df['Treatment']
    from imblearn.over_sampling import SMOTE


    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)
    # logistic regression

    LR = LogisticRegression()
    LR.fit(X_train, y_train)


    # prediction
    train_prediction= LR.predict(X_train)
    test_prediction= LR.predict(X_test)
    print('*'*20)

    # evaluation
    from sklearn.metrics import accuracy_score
    print('test accuracy:',accuracy_score(y_test,test_prediction))
    print('train accuracy:',accuracy_score(y_train,train_prediction))
    print('*'*20)

    # cross validation score
    from sklearn.model_selection import cross_val_score
    score=cross_val_score(LR,X,y,cv=5)
    print(score.mean())
    print('*'*20)


    print('*'*20)


    lr_HSC = accuracy_score(y_test,test_prediction)
    print(f"{round(lr_HSC*100,2)}% Accurate")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
    recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
    f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "Logistic Regression Algorithm"
    Logistic.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = Logistic.objects.last()
    messages.success(request, 'Algorithm executed Successfully')
    return render(request,'admin/Logistic.html',{'i':data})

def random(request):
    return render(request,'admin/randomforest.html')

def randomforest_btn(request):
    dataset = Upload_dataset_model.objects.last()
    # print(dataset.Dataset)
    df=pd.read_csv('MentalHealth_dataset\health_clean.csv')
    
    X = df.drop('Treatment', axis = 1)
    y = df['Treatment']


    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)
    from sklearn.ensemble import RandomForestClassifier

    rfc=RandomForestClassifier(random_state=42)
    rfc.fit(X_train,y_train)
    print('*'*20)

    # prediction
    train_prediction= rfc.predict(X_train)
    test_prediction= rfc.predict(X_test)
    print('*'*20)

    # evaluation
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
    recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
    f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "Random Forest Algorithm"
    RandomForest.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = RandomForest.objects.last()
    messages.success(request, 'Algorithm executed Successfully')
    request.session['accuracy']=accuracy
    # cross validation score
    print('*'*20)

    # Accuracy score
    RF_SC = accuracy_score(test_prediction,y_test)
    print(f"{round(RF_SC*100,2)}% Accurate")
    import pickle
#save the model
    model = rfc  # Your machine learning model object
    file_path = 'random.pkl'  # Path to the file where you want to save the model
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
        return render(request,'admin/randomforest.html',{'i':data})

def dtalgm(request):
    return render(request,'admin/dtalgm.html')

def Decisiontree_btn(request):
    dataset = Upload_dataset_model.objects.last()
    df=pd.read_csv('MentalHealth_dataset\health_clean.csv')
   

    X = df.drop('Treatment', axis = 1)
    y = df['Treatment']


    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)
    DT = DecisionTreeClassifier()
    DT.fit(X_train, y_train)
    from sklearn.model_selection import cross_val_score
    # y_predict = DT.predict(X_test)
    print('*'*20)

    # prediction
    train_pred=DT.predict(X_train)
    test_pred= DT.predict(X_test)
    print('*'*20)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = round(accuracy_score(y_test,test_pred)*100, 2)
    precession = round(precision_score(y_test,test_pred,average = 'macro')*100, 2)
    recall = round(recall_score(y_test,test_pred,average = 'macro')*100, 2)
    f1 = round(f1_score(y_test,test_pred,average = 'macro')*100, 2)
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "Decision Tree Algorithm"
    DECISSION_ALGO.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = DECISSION_ALGO.objects.last()
    messages.success(request, 'Algorithm executed Successfully')
    request.session['accuracy']=accuracy
    # accuracy
    print('Train accuracy:' , accuracy_score(y_train,train_pred))
    print('Test accuracy:' , accuracy_score(y_test,test_pred))

    print('*'*20)
    # cross validation   
    score= cross_val_score(DT,X,y,cv=5)
    print(score)
    print(score.mean())

    print('*'*20)
    #  prediction Summary by species

    print('*'*20)
    # Accuracy score
    DT_SC = accuracy_score(test_pred,y_test)
    print(f"{round(DT_SC*100,2)}% Accurate")

    print('*'*20)
    return render(request,'admin/dtalgm.html',{'i':data})

    
def gdalgm(request):
    return render(request,'admin/gd-boost.html')

def GD_btn(request):
    dataset = Upload_dataset_model.objects.last()
    # print(dataset.Dataset)
    df=pd.read_csv('MentalHealth_dataset\health_clean.csv')
    

    X = df.drop('Treatment', axis = 1)
    y = df['Treatment']
    from imblearn.over_sampling import SMOTE


    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)
    from sklearn.ensemble import GradientBoostingClassifier
    # random forest

    gbc=GradientBoostingClassifier(random_state=42)
    gbc.fit(X_train,y_train)
    print('*'*20)

    # prediction
    train_prediction= gbc.predict(X_train)
    test_prediction= gbc.predict(X_test)
    print('*'*20)

    # evaluation
    from sklearn.metrics import accuracy_score
    print('test accuracy:',accuracy_score(y_test,test_prediction))
    print('train accuracy:',accuracy_score(y_train,train_prediction))
    print('*'*20)

    # cross validation score
    from sklearn.model_selection import cross_val_score
    score=cross_val_score(gbc,X,y,cv=5)
    print(score.mean())
    print('*'*20)

    #  prediction Summary by species
    # print(classification_report(y_test, test_prediction))
    print('*'*20)

    # Accuracy score
    gbc_h = accuracy_score(test_prediction,y_test)
    print(f"{round(gbc_h*100,2)}% Accurate")
    # evaluation
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction)*100, 2)
    recall = round(recall_score(y_test,test_prediction)*100, 2)
    f1 = round(f1_score(y_test,test_prediction)*100, 2)
    name = "Gradient Boost Algorithm"
    
    GradientBoosting.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = GradientBoosting.objects.last()
    messages.success(request, 'Algorithm executed Successfully')


    return render(request,'admin/gd-boost.html',{'i':data})

def admin_graph(request):
    details4 = RandomForest.objects.last()
    a = details4.Accuracy

    details5 = Logistic.objects.last()
    b = details5.Accuracy

    details = XGBR.objects.last()
    c = details.Accuracy

    deatails1 = ADA_ALGO.objects.last()
    d = deatails1.Accuracy

    details2 = KNN_ALGO.objects.last()
    e = details2.Accuracy

    details9 = GradientBoosting.objects.last()
    f = details9.Accuracy

    details10 = DECISSION_ALGO.objects.last()
    z = details10.Accuracy

    details11 = BiLSTM_CNN.objects.last()
    bl = details11.Accuracy

    print(a,b, c,d,e,f,'aaaa')
    print(details4, details, deatails1,details2,details9, details5,'aaaa')
    return render(request,'admin/admin-graph.html',{'xg':c,'ada':d,'knn':e,'dt':z,'log':b, 'ran':a, 'gst': f,'BiLSTM_CNN':bl})

def user_feedbacks(request):
    feed = UserFeedbackModels.objects.all()

    return render(request,'admin/admin-feedbacks.html',{'back':feed})

def user_sentiment(request):
    feed = UserFeedbackModels.objects.all()
    return render(request,'admin/admin-sentimentanalysis.html',{'back':feed})

def user_graph(request):
    positive = UserFeedbackModels.objects.filter(sentment = 'positive').count()
    very_positive = UserFeedbackModels.objects.filter(sentment = 'very positive').count()
    negative = UserFeedbackModels.objects.filter(sentment = 'negative').count()
    very_negative = UserFeedbackModels.objects.filter(sentment = 'very negative').count()
    neutral = UserFeedbackModels.objects.filter(sentment = 'neutral').count()
    context ={
        'vp': very_positive, 'p':positive, 'n':negative, 'vn':very_negative, 'ne':neutral
    }
    return render(request,'admin/admin-feedback-graph.html',context)

def Admin_Reject_Btn(request, x):
    user=UserDetails.objects.get(user_id=x)
    user.user_status="Rejected"
    messages.success(request,"Status Changed  Successfully")

    user.save()
    messages.warning(request,"rejected")

    return redirect("admin_pendingusers")

def Admin_Accept_Button(request,x):
    user=UserDetails.objects.get(user_id=x)
    user.user_status="Accepted"
    messages.success(request,"Status Changed Successfully")

    user.save()
    messages.warning(request,"Accepted")

    return redirect("admin_pendingusers")

def Change_Status(request,id):
    # user_id=req.session["User_id"]
    user=UserDetails.objects.get(user_id=id)
    if user.user_status=="Accepted":
        user.user_status=="Rejected"
        user.save()
        messages.success(request,"Status Changed Successfully")

        return redirect("all_users")
    
    else:
        user.user_status="Accepted"
        user.save()
        messages.success(request,"Status Successfully Changed")

        return redirect("all_users")
    
def delete_User(request,id):
    UserDetails.objects.get(user_id=id).delete()
    messages.info(request,"Deleted")

    return redirect("all_users")
