from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, PoissonRegressor
from predictor.forms import BreastCancerForm, DiabeticForm, HeartDiseaseForm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from predictor.forms import pimadiabetesForm

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier

from predictor.forms import DiabeticRetinopathyForm
from predictor.utils import generate_performance_chart, get_medicine_suggestions,generate_stroke_chart,generate_Pimadiabetes_chart
from predictor.utils import generate_breast_chart,generate_diabetic_chart,generate_retinopathy_chart



def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            return render(request, 'login.html', {'error': 'Invalid username or password'})
    return render(request, 'login.html')


def heart(request):
    import warnings
    warnings.filterwarnings("ignore")

    df = pd.read_csv('static/Heart_train.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1:].ravel()

    value = ''
    all_metrics = []

    if request.method == 'POST':
        age = float(request.POST['age'])
        sex = float(request.POST['sex'])
        cp = float(request.POST['cp'])
        trestbps = float(request.POST['trestbps'])
        chol = float(request.POST['chol'])
        fbs = float(request.POST['fbs'])
        restecg = float(request.POST['restecg'])
        thalach = float(request.POST['thalach'])
        exang = float(request.POST['exang'])
        oldpeak = float(request.POST['oldpeak'])
        slope = float(request.POST['slope'])
        ca = float(request.POST['ca'])
        thal = float(request.POST['thal'])

        user_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)

        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(np.nan_to_num(X), Y, test_size=0.2, random_state=42)

        models = {
            'Random Forest': RandomForestClassifier(n_estimators=16, criterion='entropy', max_depth=9),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Linear Regression': LinearRegression(),
            'Poisson Regression': PoissonRegressor(max_iter=300),
            'SVM': SVC(probability=True),
            'Decision Tree': DecisionTreeClassifier(max_depth=5)
        }

        best_model = None
        best_score = 0
        best_model_prediction = None
        results = {}

        for name, model in models.items():
            try:
                model.fit(X_train, Y_train)
                y_pred = model.predict(X_test)

                if name in ['Linear Regression', 'Poisson Regression']:
                    y_pred = (y_pred > 0.5).astype(int)

                acc = accuracy_score(Y_test, y_pred)
                rec = recall_score(Y_test, y_pred)
                f1 = f1_score(Y_test, y_pred)

                if acc > best_score:
                    best_score = acc
                    best_model = name

                model_prediction = model.predict(user_data)

                if not isinstance(model_prediction, np.ndarray):
                    model_prediction = np.array([model_prediction])

                if name in ['Linear Regression', 'Poisson Regression']:
                    model_prediction = (model_prediction > 0.5).astype(int)

                user_result = 'have' if int(model_prediction[0]) == 1 else "don’t have"

                if name == best_model:
                    best_model_prediction = user_result

                results[name] = {
                    'prediction': user_result,
                    'accuracy': acc,
                    'recall': rec,
                    'f1_score': f1
                }

            except Exception as e:
                results[name] = {
                    'error': str(e)
                }

        chart_url = generate_performance_chart(results)
        corr = df.corr()['target'].sort_values(ascending=False)
        chart_url1 = generate_stroke_chart(corr)
        return render(request, 'result.html', {
            'context': results,
            'result': best_model_prediction,
            'best_model': best_model,
            'medicines': get_medicine_suggestions('heart','mild' if best_model_prediction == 'don’t have' else 'severe'),
            'chart': chart_url,
            'title': 'Heart Disease Prediction',
            'disease_name': 'heart disease',
            'context': results,
            'chart1': chart_url1,
            'title': 'Stroke Risk Prediction',
            'disease_name': 'stroke risk',
        })

    return render(request, 'heart.html', {
        'context': value,
        'title': 'Heart Disease Prediction',
        'active': 'btn btn-success peach-gradient text-white',
        'heart': True,
        'form': HeartDiseaseForm(),
    })



def pimadiabetes(request):
    import warnings
    warnings.filterwarnings("ignore")

    dfx = pd.read_csv('static/Diabetes_XTrain.csv')
    dfy = pd.read_csv('static/Diabetes_YTrain.csv')
    X = dfx.values
    Y = dfy.values.reshape((-1,))

    value = ''
    best_model = None
    best_score = 0
    best_model_prediction = None
    results = {}

    if request.method == 'POST':
        pregnancies = float(request.POST['pregnancies'])
        glucose = float(request.POST['glucose'])
        bloodpressure = float(request.POST['bloodpressure'])
        skinthickness = float(request.POST['skinthickness'])
        bmi = float(request.POST['bmi'])
        insulin = float(request.POST['insulin'])
        pedigree = float(request.POST['pedigree'])
        age = float(request.POST['age'])

        user_data = np.array(
            (pregnancies, glucose, bloodpressure, skinthickness, bmi, insulin, pedigree, age)
        ).reshape(1, 8)



        X_train, X_test, Y_train, Y_test = train_test_split(np.nan_to_num(X), Y, test_size=0.2, random_state=42)

        models = {
            'Random Forest': RandomForestClassifier(n_estimators=16, criterion='entropy', max_depth=9),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Linear Regression': LinearRegression(),
            'Poisson Regression': PoissonRegressor(max_iter=300),
            'SVM': SVC(probability=True),
            'Decision Tree': DecisionTreeClassifier(max_depth=5)
        }

        for name, model in models.items():
            try:
                model.fit(X_train, Y_train)
                y_pred = model.predict(X_test)

                if name in ['Linear Regression', 'Poisson Regression']:
                    y_pred = (y_pred > 0.5).astype(int)

                acc = accuracy_score(Y_test, y_pred)
                rec = recall_score(Y_test, y_pred)
                f1 = f1_score(Y_test, y_pred)

                if acc > best_score:
                    best_score = acc
                    best_model = name

                model_prediction = model.predict(user_data)
                if not isinstance(model_prediction, np.ndarray):
                    model_prediction = np.array([model_prediction])

                if name in ['Linear Regression', 'Poisson Regression']:
                    model_prediction = (model_prediction > 0.5).astype(int)

                user_result = 'have' if int(model_prediction[0]) == 1 else "don’t have"

                if name == best_model:
                    best_model_prediction = user_result

                results[name] = {
                    'prediction': user_result,
                    'accuracy': acc,
                    'recall': rec,
                    'f1_score': f1
                }

            except Exception as e:
                results[name] = {
                    'error': str(e)
                }
        chart_url = generate_performance_chart(results)
        corr = dfy.corr()['Outcome'].sort_values(ascending=False)
        chart_url1 = generate_Pimadiabetes_chart(corr)
        value = best_model_prediction
        return render(request, 'pimadiabetesresults.html', {
            'context': results,
            'result': best_model_prediction,
            'best_model': best_model,
            'medicines': get_medicine_suggestions('Pimadiabetes','controlled' if best_model_prediction == 'don’t have' else 'uncontrolled'),
            'chart': chart_url,
            'title': 'Pimadiabetes Disease Prediction',
            'disease_name': 'pimadiabetes',
            'context': results,
            'chart1': chart_url1,
            'title': 'Pimadiabetes Risk Prediction',
            'disease_name': 'Pimadiabetes risk',
        })

    return render(request, 'pimadiabetes.html', {
        'context': value,
        'title': 'pimadiabetes Disease Prediction',
        'active': 'btn btn-success peach-gradient text-white',
        'pimadiabetes': True,
        'form': pimadiabetesForm(),
    })

def breast(request):
    import warnings
    warnings.filterwarnings("ignore")

    df = pd.read_csv('static/Breast_train.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1]

    value = ''
    best_model = None
    best_score = 0
    best_model_prediction = None
    results = {}

    if request.method == 'POST':
        radius = float(request.POST['radius'])
        texture = float(request.POST['texture'])
        perimeter = float(request.POST['perimeter'])
        area = float(request.POST['area'])
        smoothness = float(request.POST['smoothness'])

        user_data = np.array((radius, texture, perimeter, area, smoothness)).reshape(1, 5)



        X_train, X_test, Y_train, Y_test = train_test_split(np.nan_to_num(X), Y, test_size=0.2, random_state=42)

        models = {
            'Random Forest': RandomForestClassifier(n_estimators=16, criterion='entropy', max_depth=9),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Linear Regression': LinearRegression(),
            'Poisson Regression': PoissonRegressor(max_iter=300),
            'SVM': SVC(probability=True),
            'Decision Tree': DecisionTreeClassifier(max_depth=5)
        }

        for name, model in models.items():
            try:
                model.fit(X_train, Y_train)
                y_pred = model.predict(X_test)

                if name in ['Linear Regression', 'Poisson Regression']:
                    y_pred = (y_pred > 0.5).astype(int)

                acc = accuracy_score(Y_test, y_pred)
                rec = recall_score(Y_test, y_pred)
                f1 = f1_score(Y_test, y_pred)

                if acc > best_score:
                    best_score = acc
                    best_model = name

                model_prediction = model.predict(user_data)
                if not isinstance(model_prediction, np.ndarray):
                    model_prediction = np.array([model_prediction])

                if name in ['Linear Regression', 'Poisson Regression']:
                    model_prediction = (model_prediction > 0.5).astype(int)

                user_result = 'have' if int(model_prediction[0]) == 1 else "don’t have"

                if name == best_model:
                    best_model_prediction = user_result

                results[name] = {
                    'prediction': user_result,
                    'accuracy': acc,
                    'recall': rec,
                    'f1_score': f1
                }

            except Exception as e:
                results[name] = {
                    'error': str(e)
                }

        from .utils import generate_performance_chart
        chart_url = generate_performance_chart(results)

        corr = df.corr()['diagnosis'].sort_values(ascending=False)
        chart_url1 = generate_breast_chart(corr)

        value = best_model_prediction
        return render(request, 'cancerresult.html', {
            'result': value,
            'best_model': best_model,
            'medicines': get_medicine_suggestions('breast', 'stage_1_2' if value == 'don’t have' else 'stage_3_4'),
            'chart': chart_url,
            'title': 'Breast Cancer Prediction',
            'disease_name': 'breast cancer',
            'chart1': chart_url1,
            'title': 'Breast Cancer Risk Prediction',
            'disease_name': 'Breast Cancer risk',
        })

    return render(request, 'breast.html', {
        'context': value,
        'title': 'Breast Cancer Prediction',
        'active': 'btn btn-success peach-gradient text-white',
        'breast': True,
        'form': BreastCancerForm(),
    })
def retinopathy(request):
    df = pd.read_csv('static/DiabeticRetinopathy.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1:].ravel()
    value = ''

    def safe_float(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    if request.method == 'POST':
        age = safe_float(request.POST.get('age'))
        diabetes_duration = safe_float(request.POST.get('diabetes_duration'))
        hba1c = safe_float(request.POST.get('hba1c'))
        blood_glucose = safe_float(request.POST.get('blood_glucose'))
        systolic_bp = safe_float(request.POST.get('systolic_bp'))
        diastolic_bp = safe_float(request.POST.get('diastolic_bp'))
        vision_blur = safe_float(request.POST.get('vision_blur'))
        eye_pain = safe_float(request.POST.get('eye_pain'))
        family_history = safe_float(request.POST.get('family_history'))
        image_severity_score = safe_float(request.POST.get('image_severity_score'))

        user_data = np.array([
            age, diabetes_duration, hba1c, blood_glucose,
            systolic_bp, diastolic_bp, vision_blur,
            eye_pain, family_history, image_severity_score
        ]).reshape(1, -1)

        X_train, X_test, Y_train, Y_test = train_test_split(
            np.nan_to_num(X), Y, test_size=0.2, random_state=42)

        models = {
            'Random Forest': RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=9),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Linear Regression': LinearRegression(),
            'Poisson Regression': PoissonRegressor(max_iter=300),
            'SVM': SVC(probability=True),
            'Decision Tree': DecisionTreeClassifier(max_depth=5)
        }

        best_model = None
        best_score = 0
        best_model_prediction = None
        results = {}

        for name, model in models.items():
            try:
                model.fit(X_train, Y_train)
                y_pred = model.predict(X_test)

                if name in ['Linear Regression', 'Poisson Regression']:
                    y_pred = (y_pred > 0.5).astype(int)

                acc = accuracy_score(Y_test, y_pred)
                rec = recall_score(Y_test, y_pred)
                f1 = f1_score(Y_test, y_pred)

                model_prediction = model.predict(user_data)
                if name in ['Linear Regression', 'Poisson Regression']:
                    model_prediction = (model_prediction > 0.5).astype(int)

                user_result = 'have' if int(model_prediction[0]) == 1 else "don’t have"

                if acc > best_score:
                    best_score = acc
                    best_model = name
                    best_model_prediction = user_result

                results[name] = {
                    'prediction': user_result,
                    'accuracy': acc,
                    'recall': rec,
                    'f1_score': f1
                }

            except Exception as e:
                # Skip model from results if it fails
                continue

        chart_url = generate_performance_chart(results)

        corr = df.corr()['diagnosis'].sort_values(ascending=False)
        chart_url1 = generate_retinopathy_chart(corr)
        return render(request, 'result.html', {
            'result': best_model_prediction,
            'best_model': best_model,
            'medicines': get_medicine_suggestions(
                'retinopathy',
                'mild' if best_model_prediction == 'don’t have' else 'severe'
            ),
            'chart': chart_url,
            'title': 'Diabetic Retinopathy Prediction',
            'disease_name': 'diabetic retinopathy',
            'chart1': chart_url1,
            'title': 'Diabetic Retinopathy Risk Prediction',
            'disease_name': 'Diabetic Retinopathy risk',
        })

    return render(request, 'retinopathy.html', {
        'context': value,
        'title': 'Diabetic Retinopathy Prediction',
        'active': 'btn btn-success peach-gradient text-white',
        'retinopathy': True,
        'form': DiabeticRetinopathyForm(),
    })

def diabetic(request):
    df = pd.read_csv('static/diabetic.csv')  # Ensure dataset matches form fields
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1:].ravel()

    value = ''
    if request.method == 'POST':
        def safe_float(value):
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0

        # Collect user input
        age = safe_float(request.POST.get('age'))
        gender = safe_float(request.POST.get('gender'))
        bmi = safe_float(request.POST.get('bmi'))
        blood_glucose = safe_float(request.POST.get('blood_glucose'))
        hba1c = safe_float(request.POST.get('hba1c'))
        systolic_bp = safe_float(request.POST.get('systolic_bp'))
        diastolic_bp = safe_float(request.POST.get('diastolic_bp'))
        cholesterol = safe_float(request.POST.get('cholesterol'))
        family_history = safe_float(request.POST.get('family_history'))
        physical_activity = safe_float(request.POST.get('physical_activity'))

        user_data = np.array([
            age, gender, bmi, blood_glucose, hba1c,
            systolic_bp, diastolic_bp, cholesterol,
            family_history, physical_activity
        ]).reshape(1, -1)

        # Train-test split
        X_train, X_test, Y_train, Y_test = train_test_split(
            np.nan_to_num(X), Y, test_size=0.2, random_state=42)

        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=8),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Linear Regression': LinearRegression(),
            'Poisson Regression': PoissonRegressor(max_iter=300),
            'SVM': SVC(probability=True),
            'Decision Tree': DecisionTreeClassifier(max_depth=5)
        }

        best_model = None
        best_score = 0
        best_model_prediction = None
        results = {}

        for name, model in models.items():
            try:
                model.fit(X_train, Y_train)
                y_pred = model.predict(X_test)

                # Convert regression output to binary
                if name in ['Linear Regression', 'Poisson Regression']:
                    y_pred = (y_pred > 0.5).astype(int)

                acc = accuracy_score(Y_test, y_pred)
                rec = recall_score(Y_test, y_pred)
                f1 = f1_score(Y_test, y_pred)

                # Predict for user input
                model_prediction = model.predict(user_data)
                if name in ['Linear Regression', 'Poisson Regression']:
                    model_prediction = (model_prediction > 0.5).astype(int)

                user_result = 'have' if int(model_prediction[0]) == 1 else "don’t have"

                # Track best performing model
                if acc > best_score:
                    best_score = acc
                    best_model = name
                    best_model_prediction = user_result

                # Store performance
                results[name] = {
                    'prediction': user_result,
                    'accuracy': acc,
                    'recall': rec,
                    'f1_score': f1
                }

            except Exception as e:
                # Ignore failed models (do not add to results)
                continue

        # Generate chart from only successful models
        chart_url = generate_performance_chart(results)
        corr = df.corr()['diagnosis'].sort_values(ascending=False)
        chart_url1 = generate_diabetic_chart(corr)

        return render(request, 'result.html', {
            'result': best_model_prediction,
            'best_model': best_model,
            'medicines': get_medicine_suggestions(
                'diabetes', 'mild' if best_model_prediction == 'don’t have' else 'severe'
            ),
            'chart': chart_url,
            'title': 'Diabetic Prediction',
            'disease_name': 'diabetic',
            'chart1': chart_url1,
            'title': 'Diabetic Risk Prediction',
            'disease_name': 'Diabetic risk',
        })

    # Initial form render
    return render(request, 'diabetic.html', {
        'context': value,
        'title': 'Diabetic Prediction',
        'active': 'btn btn-success peach-gradient text-white',
        'diabetic': True,
        'form': DiabeticForm(),
    })

def home(request):
    return render(request, 'home.html')


# ✅ Correct
from django.shortcuts import render

def handler404(request, exception):
    return render(request, '404.html', status=404)

