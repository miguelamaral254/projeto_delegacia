# train_baseline.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from nltk.corpus import stopwords
import nltk

print("--- Módulo de Treinamento: Baseline (DummyClassifier) ---")

# --- Download de dependências do NLTK ---
try:
    stopwords.words('portuguese')
except LookupError:
    print("Baixando o pacote 'stopwords' do NLTK...")
    nltk.download('stopwords')

# --- DEFINIÇÃO DOS CAMINHOS ---
# Assume que este script está no diretório 'src'
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_FILE = BASE_DIR / "data" / "dataset_ocorrencias_delegacia_5.csv"
REPORTS_DIR = BASE_DIR / "reports"

os.makedirs(REPORTS_DIR, exist_ok=True)

# --- Carregamento e Preparação dos Dados ---
df = pd.read_csv(DATA_FILE)
df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'])

target = "tipo_crime"
features_to_drop = [target, "id_ocorrencia", "data_ocorrencia"]

# --- Divisão Temporal (Temporal Split) ---
df_sorted = df.sort_values("data_ocorrencia")
train_size = int(0.8 * len(df_sorted))
train_df = df_sorted.iloc[:train_size]
test_df = df_sorted.iloc[train_size:]

X_train, y_train = train_df.drop(columns=features_to_drop), train_df[target]
X_test, y_test = test_df.drop(columns=features_to_drop), test_df[target]

# --- Definição das colunas ---
num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
text_col = 'descricao_modus_operandi'
cat_cols = [col for col in X_train.select_dtypes(include=['object']).columns.tolist() if col != text_col]

# --- Pipeline de Pré-processamento ---
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("text_tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words=stopwords.words('portuguese')), text_col)
    ],
    remainder='passthrough'
)

# --- Pipeline de Modelagem (Baseline) ---
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DummyClassifier(strategy='most_frequent', random_state=42))
])

# Treinamento
print("Treinando o modelo Baseline...")
pipeline.fit(X_train, y_train)

# Predição e Avaliação
print("Avaliando o modelo Baseline...")
y_pred = pipeline.predict(X_test)
print("Relatório de Classificação (Baseline):")
print(classification_report(y_test, y_pred, zero_division=0))

# Salvar Matriz de Confusão
fig, ax = plt.subplots(figsize=(15, 15))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, xticks_rotation='vertical', colorbar=True)
plt.title('Matriz de Confusão - Baseline (Dummy)')
plt.tight_layout()
confusion_matrix_path = REPORTS_DIR / "confusion_matrix_Baseline.png"
plt.savefig(confusion_matrix_path)

print(f"Matriz de Confusão do Baseline salva em: {confusion_matrix_path}")
print("--- Módulo Baseline concluído. ---")