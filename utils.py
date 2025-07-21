# predictor/utils.py
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64

def get_medicine_suggestions(disease_type, severity):
    """
    Returns sample medicine suggestions based on disease type and severity.
    """
    medicine_data = {
        'heart': {
            'mild': ['Aspirin', 'Atorvastatin'],
            'severe': ['Nitroglycerin', 'Beta-blockers']
        },
        'diabetes': {
            'mild': ['Metformin', 'Glipizide'],
            'severe': ['Insulin', 'SGLT2 Inhibitors']
        },
        'breast': {
            'stage_1_2': ['Tamoxifen', 'Letrozole'],
            'stage_3_4': ['Chemotherapy', 'Targeted Therapy']
        },
        'retinopathy': {
            'mild': ['Control Blood Sugar', 'Regular Eye Exams'],
            'severe': ['Anti-VEGF Injections', 'Laser Treatment']
        },
        'Pimadiabetes': {
        'controlled': [
            'Metformin',
            'Low-Carb Diet',
            '30 Minutes Daily Exercise',
            'Quarterly HbA1c Monitoring'
        ],
        'uncontrolled': [
            'Insulin Therapy',
            'SGLT2 or GLP-1 Agonists',
            'Continuous Glucose Monitoring (CGM)',
            'Consult Endocrinologist'
        ]
        }
    }
    return medicine_data.get(disease_type, {}).get(severity, ['Consult Doctor'])

def generate_performance_chart(results):
    models = list(results.keys())
    accuracy = [results[m]['accuracy'] for m in models]
    recall = [results[m]['recall'] for m in models]
    f1_score = [results[m]['f1_score'] for m in models]
    bar_width = 0.25
    x = range(len(models))
    plt.figure(figsize=(10, 6))
    plt.bar(x, accuracy, width=bar_width, label='Accuracy', color='skyblue')
    plt.bar([p + bar_width for p in x], recall, width=bar_width, label='Recall', color='orange')
    plt.bar([p + 2 * bar_width for p in x], f1_score, width=bar_width, label='F1 Score', color='green')
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Model Performance Comparison')
    plt.xticks([p + bar_width for p in x], models, rotation=45)
    plt.legend()
    plt.tight_layout()
    # Save to memory buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    # Encode as base64
    graph = base64.b64encode(image_png).decode('utf-8')
    return f"data:image/png;base64,{graph}"

import os
import matplotlib.pyplot as plt
import seaborn as sns
import uuid

def generate_stroke_chart(results):
    if isinstance(results, dict):
        import pandas as pd
        results = pd.Series(results)
    folder_path = 'static/images'
    os.makedirs(folder_path, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=results.index, y=results.values, palette='coolwarm')
    plt.title("Stroke Correlation with Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = f"stroke_chart_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(folder_path, filename)
    plt.savefig(filepath)
    plt.close()
    return f"/static/images/{filename}"

def generate_Pimadiabetes_chart(results):
    if isinstance(results, dict):
        import pandas as pd
        results = pd.Series(results)
    folder_path = 'static/images'
    os.makedirs(folder_path, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=results.index, y=results.values, palette='coolwarm')
    plt.title("Pimadiabetes Correlation with Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = f"stroke_chart_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(folder_path, filename)
    plt.savefig(filepath)
    plt.close()
    return f"/static/images/{filename}"

def generate_breast_chart(results):
    if isinstance(results, dict):
        import pandas as pd
        results = pd.Series(results)
    folder_path = 'static/images'
    os.makedirs(folder_path, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=results.index, y=results.values, palette='coolwarm')
    plt.title("Brest Cancer Correlation with Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = f"stroke_chart_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(folder_path, filename)
    plt.savefig(filepath)
    plt.close()
    return f"/static/images/{filename}"

def generate_diabetic_chart(results):
    if isinstance(results, dict):
        import pandas as pd
        results = pd.Series(results)
    folder_path = 'static/images'
    os.makedirs(folder_path, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=results.index, y=results.values, palette='coolwarm')
    plt.title("Diabetic Correlation with Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = f"Diabetic_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(folder_path, filename)
    plt.savefig(filepath)
    plt.close()
    return f"/static/images/{filename}"

def generate_retinopathy_chart(results):
    if isinstance(results, dict):
        import pandas as pd
        results = pd.Series(results)
    folder_path = 'static/images'
    os.makedirs(folder_path, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=results.index, y=results.values, palette='coolwarm')
    plt.title("Diabetic retinopathy Correlation with Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = f"Diabeticretinopathy_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(folder_path, filename)
    plt.savefig(filepath)
    plt.close()
    return f"/static/images/{filename}"