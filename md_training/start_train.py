import os
import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
STATUS_FILE = REPORTS_DIR / "training_status.json"

def write_status(status, message=None):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    data = {"status": status, "timestamp": datetime.now().isoformat()}
    if message:
        data["message"] = message
    with open(STATUS_FILE, 'w') as f:
        json.dump(data, f)

def run_script(script_path):
    script_name = script_path.name
    print(f"--- Iniciando a execução de {script_name} ---")
    
    # O check=True fará o processo falhar se o script interno falhar (status 1)
    process = subprocess.run(
        [sys.executable, str(script_path)],
        check=True, 
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    # NÃO IMPRIMIR process.stdout AQUI. O script de treino já deve imprimir seus próprios resultados.
    
    print(f"--- {script_name} concluído com sucesso! ---\n")

if __name__ == "__main__":
    try:
        write_status("running", "Pipeline de treinamento iniciado.")
        print(">>> Iniciando o pipeline de treinamento de modelos...")

        current_dir = Path(__file__).resolve().parent
        # A ordem é importante, pois o LightGBM é o modelo principal de predição de violência
        training_scripts = ["train_baseline.py", "train_randomforest.py", "train_lightgbm.py"]

        for script_filename in training_scripts:
            full_script_path = current_dir / script_filename
            if not full_script_path.exists():
                raise FileNotFoundError(f"O script {script_filename} não foi encontrado.")
            run_script(full_script_path)
        
        write_status("complete", "Pipeline de treinamento concluído com sucesso.")
        print(">>> Pipeline de treinamento de modelos finalizado.")

    except Exception as e:
        error_message = f"ERRO: {e}"
        print(error_message)
        write_status("failed", error_message)
        sys.exit(1)