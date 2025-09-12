# main.py (Corrigido)
import os
import subprocess
import sys
from pathlib import Path # NOVO: Importa a classe Path

def run_script(script_path):
    """Executa um script Python usando seu caminho completo e verifica se houve erros."""
    script_name = script_path.name
    print(f"--- Iniciando a execução de {script_name} ---")
    try:
        process = subprocess.run(
            [sys.executable, str(script_path)], # Garante que o caminho é uma string
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8' # Adicionado para melhor compatibilidade
        )
        print(process.stdout)
        print(f"--- {script_name} concluído com sucesso! ---\n")
    except subprocess.CalledProcessError as e:
        print(f"!!! ERRO ao executar {script_name}:")
        print(e.stderr)
        print("!!! A execução será interrompida. Corrija o erro acima e tente novamente.")
        sys.exit(1) # Interrompe a execução em caso de erro

if __name__ == "__main__":
    print(">>> Iniciando o pipeline de treinamento de modelos...")

    # NOVO: Obtém o diretório do script atual (main.py)
    # Isso garante que ele sempre encontrará os outros scripts na mesma pasta.
    current_dir = Path(__file__).resolve().parent

    training_scripts = [
        "train_baseline.py",
        "train_randomforest.py",
        "train_lightgbm.py"
    ]

    for script_filename in training_scripts:
        # NOVO: Monta o caminho completo para o script
        full_script_path = current_dir / script_filename
        if not full_script_path.exists():
            print(f"!!! ERRO: O script {script_filename} não foi encontrado no diretório {current_dir}")
            continue
        run_script(full_script_path)

    print(">>> Pipeline de treinamento de modelos finalizado.")