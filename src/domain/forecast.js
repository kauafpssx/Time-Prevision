export function formatForecastDate(value, options = {}) {
  return new Intl.DateTimeFormat('pt-BR', { timeZone: 'UTC', ...options })
    .format(new Date(`${value}T12:00:00Z`));
}

export function displayValue(value, fallback = '--') {
  return value ?? fallback;
}

export function visibleDays(forecast, showAll) {
  return showAll ? forecast.daily : forecast.daily.slice(0, 7);
}

export function validateForecast(forecast) {
  if (!forecast?.daily?.length || !forecast?.hourly?.length) {
    throw new Error('Formato da previsão inválido');
  }
  return forecast;
}
