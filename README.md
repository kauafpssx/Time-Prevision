# Time Prevision

Previsão meteorológica experimental para Porto Alegre. O backend Python roda diariamente no GitHub Actions e publica um frontend estático em HTML, CSS e JavaScript puro.

O modelo treinado tem cerca de 177 MB e usa Git LFS. Antes do primeiro commit, execute `git lfs install`.

## Arquitetura

O código versionado fica em apenas duas pastas principais:

```text
backend/
├── data/                         # banco e artefatos do modelo
└── src/time_prevision/
    ├── domain/                   # regras puras do clima
    ├── application/              # casos de uso
    ├── infrastructure/           # banco, ML e Open-Meteo
    └── interfaces/cli/           # comandos de entrada

frontend/
├── assets/css/                   # estilos
├── data/                         # JSON publicado
└── src/
    ├── domain/                   # validação e formatação
    ├── application/              # casos de uso
    ├── infrastructure/           # acesso ao forecast.json
    └── presentation/             # DOM e inicialização
```

As dependências apontam para dentro: apresentação/CLI chama aplicação, aplicação usa contratos e regras, e detalhes externos ficam na infraestrutura.

## Execução local

No PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
$env:PYTHONPATH = "backend\src"
python -m time_prevision.interfaces.cli.export_forecast --dias 30
cd frontend
python -m http.server 8000
```

Abra `http://localhost:8000`.

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

O workflow `.github/workflows/daily-forecast.yml` roda às **06:17 no horário de Brasília** em Linux com 4 CPUs. A cada execução ele atualiza o banco, retreina todos os horizontes usando o cache de hiperparâmetros, gera um JSON com os outputs e métricas do modelo recém-treinado e publica apenas `frontend/` na branch órfã `frontend`.

No GitHub:

1. Permita escrita em **Settings → Actions → General → Workflow permissions**.
2. Em **Settings → Pages**, selecione **Deploy from a branch**.
3. Selecione `frontend` e `/(root)`.
4. Execute manualmente o workflow uma vez para criar a branch.

As previsões são experimentais e não substituem alertas meteorológicos oficiais.
