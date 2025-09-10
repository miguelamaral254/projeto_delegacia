import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from nltk.corpus import stopwords
import nltk

try:
    stopwords.words('portuguese')
except LookupError:
    print("Baixando o pacote 'stopwords' do NLTK...")
    nltk.download('stopwords')

# --- CORREÇÃO DEFINITIVA DOS CAMINHOS ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_FILE = BASE_DIR / "data" / "dataset_ocorrencias_delegacia_5.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_PATH = ARTIFACTS_DIR / "lgbm_model.joblib"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)

df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'])
df['ano'] = df['data_ocorrencia'].dt.year
df['mes'] = df['data_ocorrencia'].dt.month
df['dia_semana'] = df['data_ocorrencia'].dt.dayofweek
df['hora'] = df['data_ocorrencia'].dt.hour

portuguese_stopwords = stopwords.words('portuguese')

target = "tipo_crime"
features_to_drop = [target, "id_ocorrencia", "data_ocorrencia"]

X = df.drop(columns=features_to_drop)
y = df[target]

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

text_col = 'descricao_modus_operandi'
other_cat_cols = [col for col in cat_cols if col != text_col]

df_sorted = df.sort_values("data_ocorrencia")
train_size = int(0.8 * len(df_sorted))
train_df = df_sorted.iloc[:train_size]
test_df = df_sorted.iloc[train_size:]

X_train, y_train = train_df.drop(columns=features_to_drop), train_df[target]
X_test, y_test = test_df.drop(columns=features_to_drop), test_df[target]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), other_cat_cols),
        ("text_tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words=portuguese_stopwords), text_col)
    ],
    remainder='passthrough'
)

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LGBMClassifier(random_state=42, n_jobs=-1))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("Relatório de Classificação (LightGBM):")
print(classification_report(y_test, y_pred))

fig, ax = plt.subplots(figsize=(15, 15))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, xticks_rotation='vertical', colorbar=True)
plt.title('Matriz de Confusão - LightGBM')
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH)

api_pipeline = Pipeline(steps=[
    ('preprocessor', pipeline.named_steps['preprocessor']),
    ('classifier', pipeline.named_steps['classifier'])
])

joblib.dump(api_pipeline, MODEL_PATH)

print(f"\nModelo salvo em: {MODEL_PATH}")
print(f"Matriz de Confusão salva em: {CONFUSION_MATRIX_PATH}")
print("\nProcesso concluído.")