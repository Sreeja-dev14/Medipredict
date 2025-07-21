from django import forms


class BreastCancerForm(forms.Form):

    radius = forms.FloatField(label='Mean Radius', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    texture = forms.FloatField(label='Mean Texture', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    perimeter = forms.FloatField(label='Mean Perimeter', min_value=0, max_value=300, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    area = forms.FloatField(label='Mean Area', min_value=0, max_value=1200, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    smoothness = forms.FloatField(label='Mean Smoothness', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))


class pimadiabetesForm(forms.Form):

    pregnancies = forms.FloatField(label='Pregnancies', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    glucose = forms.FloatField(label='Glucose', min_value=0, max_value=300, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    bloodpressure = forms.FloatField(label='Blood Pressure', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    skinthickness = forms.FloatField(label='Skin Thickness', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    insulin = forms.FloatField(label='Insulin', min_value=0, max_value=400, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    bmi = forms.FloatField(label='BMI', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    pedigree = forms.FloatField(label='Pedigree', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    age = forms.FloatField(label='Age', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))


class HeartDiseaseForm(forms.Form):

    age = forms.FloatField(label='Age', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    sex = forms.FloatField(label='Sex', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    cp = forms.FloatField(label='CP', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    trestbps = forms.FloatField(label='TRESTBPS', min_value=0, max_value=300, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    chol = forms.FloatField(label='CHOL', min_value=0, max_value=300, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    fbs = forms.FloatField(label='FBS', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    restecg = forms.FloatField(label='RESTECG', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    thalach = forms.FloatField(label='THALACH', min_value=0, max_value=300, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    exang = forms.FloatField(label='EXANG', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    oldpeak = forms.FloatField(label='OLDPEAK', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    slope = forms.FloatField(label='SLOPE', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    ca = forms.FloatField(label='CA', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    thal = forms.FloatField(label='THAL', min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))

from django import forms

class DiabeticForm(forms.Form):
    age = forms.FloatField(label='Age', min_value=0, max_value=120, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    gender = forms.ChoiceField(label='Gender', choices=[(0, 'Female'), (1, 'Male')], widget=forms.Select(attrs={'class': 'form-control'}))
    bmi = forms.FloatField(label='BMI', min_value=10, max_value=60, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    blood_glucose = forms.FloatField(label='Blood Glucose (mg/dL)', min_value=50, max_value=500, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    hba1c = forms.FloatField(label='HbA1c (%)', min_value=4, max_value=15, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    systolic_bp = forms.FloatField(label='Systolic BP (mmHg)', min_value=70, max_value=200, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    diastolic_bp = forms.FloatField(label='Diastolic BP (mmHg)', min_value=40, max_value=130, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    cholesterol = forms.FloatField(label='Cholesterol (mg/dL)', min_value=100, max_value=400, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    family_history = forms.ChoiceField(label='Family History of Diabetes', choices=[(0, 'No'), (1, 'Yes')], widget=forms.Select(attrs={'class': 'form-control'}))
    physical_activity = forms.ChoiceField(label='Physical Activity Level', choices=[(0, 'Low'), (1, 'Moderate'), (2, 'High')], widget=forms.Select(attrs={'class': 'form-control'}))




class DiabeticRetinopathyForm(forms.Form):
    age = forms.FloatField(label='Age',min_value=0,max_value=120,widget=forms.NumberInput(attrs={'class': 'form-control'}))
    diabetes_duration = forms.FloatField(label='Duration of Diabetes (Years)',min_value=0,max_value=50,widget=forms.NumberInput(attrs={'class': 'form-control'}))
    hba1c = forms.FloatField(label='HbA1c (%)',min_value=3,max_value=15,widget=forms.NumberInput(attrs={'class': 'form-control'}))
    blood_glucose = forms.FloatField(label='Blood Glucose (mg/dL)',min_value=50,max_value=500,widget=forms.NumberInput(attrs={'class': 'form-control'}))
    systolic_bp = forms.FloatField(label='Systolic Blood Pressure (mm Hg)',min_value=80,max_value=250,widget=forms.NumberInput(attrs={'class': 'form-control'}))
    diastolic_bp = forms.FloatField(label='Diastolic Blood Pressure (mm Hg)',min_value=40,max_value=150,widget=forms.NumberInput(attrs={'class': 'form-control'}))
    vision_blur = forms.ChoiceField(label='Experiencing Vision Blur?',choices=[(1, 'Yes'), (0, 'No')],widget=forms.Select(attrs={'class': 'form-control'}))
    eye_pain = forms.ChoiceField(label='Eye Pain or Discomfort?',choices=[(1, 'Yes'), (0, 'No')],widget=forms.Select(attrs={'class': 'form-control'}))
    family_history = forms.ChoiceField(label='Family History of Retinopathy',choices=[(1, 'Yes'), (0, 'No')],widget=forms.Select(attrs={'class': 'form-control'}))
    image_severity_score = forms.FloatField(label='Retinal Image Severity Score (Optional)',required=False,min_value=0,max_value=5,widget=forms.NumberInput(attrs={'class': 'form-control'}))
