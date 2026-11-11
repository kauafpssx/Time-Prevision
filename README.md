# Time Prevision

Backend de previsão meteorológica experimental para Porto Alegre. O Python roda diariamente no GitHub Actions; o site estático vive exclusivamente na branch `frontend`.

O modelo treinado tem cerca de 177 MB e usa Git LFS. Antes do primeiro commit, execute `git lfs install`.

## Arquitetura

Na branch `main` fica somente o backend e a automação:

```text
backend/
├── data/                         # banco e artefatos do modelo
└── src/time_prevision/
    ├── domain/                   # regras puras do clima
    ├── application/              # casos de uso
    ├── infrastructure/           # banco, ML e Open-Meteo
    └── interfaces/cli/           # comandos de entrada
```

O HTML, CSS, JavaScript e `data/forecast.json` ficam somente na branch órfã `frontend`.

## Execução local

No PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
$env:PYTHONPATH = "backend\src"
python -m time_prevision.interfaces.cli.export_forecast --dias 30 --output forecast.json
```

Atualizar a base meteorológica:

```powershell
$env:PYTHONPATH = "backend\src"
python -m time_prevision.infrastructure.database.update_database
```

Treinar novamente o modelo:

```powershell
$env:PYTHONPATH = "backend\src"
python -m time_prevision.infrastructure.ml.train_model
```

## GitHub Actions e Pages

O workflow `.github/workflows/daily-forecast.yml` roda às **06:17 no horário de Brasília** em Linux com 4 CPUs. A cada execução ele atualiza o banco, retreina todos os horizontes e abre a branch `frontend` em um worktree separado. Depois, altera e commita **somente** `data/forecast.json`; HTML, CSS e JavaScript permanecem intactos.

No GitHub:

1. Permita escrita em **Settings → Actions → General → Workflow permissions**.
2. Em **Settings → Pages**, selecione **Deploy from a branch**.
3. Selecione `frontend` e `/(root)`.
4. Execute manualmente o workflow uma vez para criar a branch.

As previsões são experimentais e não substituem alertas meteorológicos oficiais.
