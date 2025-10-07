import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
import nltk
import matplotlib.pyplot as plt

import json

from pathlib import Path



# --- FUNÇÃO FINAL CORRIGIDA: update_models_summary ---
def update_models_summary(model_id, report_data, reports_dir):
    SUMMARY_FILE = reports_dir / "models_summary.json"
    
    if SUMMARY_FILE.exists():
        with open(SUMMARY_FILE, 'r') as f:
            models_summary = json.load(f)
    else:
        models_summary = []

    accuracy = report_data['metrics']['accuracy']
    
    # CORREÇÃO CRUCIAL AQUI: Usamos o nome de arquivo EXATO que foi salvo
    # E garantimos que o nome de arquivo de relatório JSON está correto.
    report_filename_json = f"{model_id}_report.json"
    
    new_entry = {
        "model_id": model_id,
        "name": report_data['model_name'],
        "accuracy": accuracy,
        # A URL na API é '/reports/model/{report_name}'.
        # O nome do arquivo a ser passado é: 'baseline_report.json'
        "report_file": report_filename_json, 
        "training_date": report_data['training_date']
    }

    models_summary = [entry for entry in models_summary if entry["model_id"] != model_id]
    models_summary.append(new_entry)

    with open(SUMMARY_FILE, 'w') as f:
        json.dump(models_summary, f, indent=4)
    
    print(f"Resumo global de modelos atualizado ({model_id}): {SUMMARY_FILE}")
try:
    stopwords = nltk.corpus.stopwords.words('portuguese')
except LookupError:
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('portuguese')

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_FILE = BASE_DIR / "data" / "dataset_ocorrencias_delegacia_5.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)
df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'])

violent_crimes = ['Homicídio', 'Latrocínio', 'Roubo', 'Estupro', 'Sequestro', 'Violência Doméstica']
df['crime_violento'] = df['tipo_crime'].isin(violent_crimes)

target = "crime_violento"
features_to_drop = [target, "tipo_crime", "id_ocorrencia", "data_ocorrencia"]

df_sorted = df.sort_values("data_ocorrencia")
train_size = int(0.8 * len(df_sorted))
train_df = df_sorted.iloc[:train_size]
test_df = df_sorted.iloc[train_size:]

X_train, y_train = train_df.drop(columns=features_to_drop), train_df[target]
X_test, y_test = test_df.drop(columns=features_to_drop), test_df[target]

num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
text_col = 'descricao_modus_operandi'
cat_cols = [col for col in X_train.select_dtypes(include=['object']).columns.tolist() if col != text_col]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("text_tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words=stopwords), text_col)
    ],
    remainder='passthrough'
)

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LGBMClassifier(random_state=42, n_jobs=-1))
])

print("Treinando o modelo de previsão de violência...")
pipeline.fit(X_train, y_train)

print("Avaliando o modelo...")
y_pred = pipeline.predict(X_test)


report_dict = classification_report(y_test, y_pred, target_names=['Não Violento', 'Violento'], output_dict=True)

model_id = "lightgbm_violence"
model_name = "Previsão de Violência (LightGBM)"
report_filename = f"{model_id}_report.json"
image_filename = f"confusion_matrix_{model_id}.png"

fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, display_labels=['Não Violento', 'Violento'], cmap='Blues')
plt.title(f'Matriz de Confusão - {model_name}')
confusion_matrix_path = REPORTS_DIR / image_filename
plt.savefig(confusion_matrix_path)

training_report = {
    "model_name": model_name,
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "metrics": {
        "nao_violento": report_dict['Não Violento'],
        "violento": report_dict['Violento'],
        "accuracy": report_dict['accuracy'],
        "macro_avg": report_dict['macro avg'],
        "weighted_avg": report_dict['weighted avg']
    },
    "confusion_matrix_path": f"/reports/{image_filename}"
}
report_json_path = REPORTS_DIR / report_filename
with open(report_json_path, 'w') as f:
    json.dump(training_report, f, indent=4)
print(f"Relatório de treino salvo em: {report_json_path}")

update_models_summary(model_id, training_report, REPORTS_DIR)

api_pipeline = Pipeline(steps=[
    ('preprocessor', pipeline.named_steps['preprocessor']),
    ('classifier', pipeline.named_steps['classifier'])
])
model_path = ARTIFACTS_DIR / "violence_predictor_model.joblib"
joblib.dump(api_pipeline, model_path)
print(f"Modelo salvo em: {model_path}")