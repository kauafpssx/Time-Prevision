from time_prevision.infrastructure.ml.prediction_service import (
    carregar_modelo,
    gerar_previsao_horaria,
    prever_rapido,
    status_dados,
    verificar_arquivos,
)


class MlForecastGateway:
    """Adapter between the application port and the persisted ML model."""

    def __init__(self):
        self._artifacts = None

    def _load(self):
        if self._artifacts is None:
            self._artifacts = carregar_modelo()
        return self._artifacts

    def verify(self):
        return verificar_arquivos()

    def predict(self, days):
        model, features, _ = self._load()
        return prever_rapido(model, features, dias=days)

    def data_status(self):
        return status_dados()

    def hourly(self, daily_row):
        return gerar_previsao_horaria(daily_row)

    def metrics(self):
        return self._load()[2] or {}
