"""Rules that do not depend on pandas, databases, APIs or ML frameworks."""

DRY_DAY_THRESHOLD = 40


def classify_weather(precipitation_mm: float, rain_chance: float) -> tuple[str, str]:
    precipitation_mm = precipitation_mm or 0
    rain_chance = rain_chance or 0
    if rain_chance < DRY_DAY_THRESHOLD:
        return "Sem Chuva", "☀️"
    if precipitation_mm < 2.5:
        return "Garoa", "🌦️"
    if precipitation_mm < 10:
        return "Chuva Leve", "🌧️"
    if precipitation_mm < 50:
        return "Chuva Moderada", "🌧️"
    return "Chuva Forte", "⛈️"
