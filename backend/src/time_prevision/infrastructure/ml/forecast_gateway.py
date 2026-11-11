from time_prevision.infrastructure.ml.prediction_service import (
    carregar_modelo,
    gerar_previsao_horaria,
    prever_rapido,
    status_dados,
    verificar_arquivos,
)


class MlForecastGateway:
    """Adapter between the application port and the persisted ML model."""

    def verify(self):
        return verificar_arquivos()

    def predict(self, days):
        model, features, _ = carregar_modelo()
        return prever_rapido(model, features, dias=days)

    def data_status(self):
        return status_dados()

    def hourly(self, daily_row):
        return gerar_previsao_horaria(daily_row)
