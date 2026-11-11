import { validateForecast } from '../domain/forecast.js?v=20261111-2';

export class ForecastRepository {
  constructor(url = './data/forecast.json') {
    this.url = url;
  }

  async getCurrent() {
    const separator = this.url.includes('?') ? '&' : '?';
    const freshUrl = `${this.url}${separator}v=${Date.now()}`;
    const response = await fetch(freshUrl, {
      cache: 'no-store',
      headers: { 'Cache-Control': 'no-cache' },
    });
    if (!response.ok) throw new Error('Dados indisponíveis');
    return validateForecast(await response.json());
  }
}
