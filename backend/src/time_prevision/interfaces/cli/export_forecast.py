import argparse
from pathlib import Path

from time_prevision.application.generate_forecast import exportar
from time_prevision.infrastructure.ml.forecast_gateway import MlForecastGateway
from time_prevision.infrastructure.paths import PROJECT_ROOT

DEFAULT_OUTPUT = PROJECT_ROOT / "frontend" / "data" / "forecast.json"


def main():
    parser = argparse.ArgumentParser(description="Exporta previsão para o frontend estático")
    parser.add_argument("--dias", type=int, default=30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    exportar(MlForecastGateway(), args.output, args.dias)


if __name__ == "__main__":
    main()
