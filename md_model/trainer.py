# md_model/trainer.py
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

from md_core.config import DATA_PATH, MODEL_PATH, PREPROCESSOR_PATH, TARGET
from md_data_processing.preprocessor import load_data, create_preprocessor


def train_and_save_model():
    """Função completa para treinar, balancear e salvar o modelo e o pré-processador."""
    print("Iniciando o processo de treinamento...")

    # 1. Carregar e preparar os dados
    print(f"Carregando dados de: {DATA_PATH}")
    df = load_data(DATA_PATH)

    # Split temporal simples (como no seu notebook)
    df_sorted = df.sort_values("data_ocorrencia")
    train_df = df_sorted.iloc[:int(0.8 * len(df_sorted))]

    X_train = train_df.drop(columns=[TARGET, "id_ocorrencia", "data_ocorrencia"])
    y_train = train_df[TARGET]

    # 2. Criar e treinar o pré-processador
    print("Criando e treinando o pré-processador...")
    preprocessor = create_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)

    # 3. Balancear os dados com SMOTE
    print("Balanceando os dados de treino com SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_transformed, y_train)
    print("Balanceamento concluído.")

    # 4. Treinar o modelo final
    print("Treinando o modelo RandomForest...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_bal, y_train_bal)
    print("Treinamento do modelo concluído.")

    # 5. Salvar os artefatos (pré-processador e modelo)
    print(f"Salvando pré-processador em: {PREPROCESSOR_PATH}")
    dump(preprocessor, PREPROCESSOR_PATH)

    print(f"Salvando modelo em: {MODEL_PATH}")
    dump(model, MODEL_PATH)

    print("\n✅ Processo de treinamento finalizado com sucesso!")