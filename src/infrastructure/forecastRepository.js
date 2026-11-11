import { validateForecast } from '../domain/forecast.js';

export class ForecastRepository {
  constructor(url = './data/forecast.json') {
    this.url = url;
  }

  async getCurrent() {
    const response = await fetch(this.url, { cache: 'no-store' });
    if (!response.ok) throw new Error('Dados indisponíveis');
    return validateForecast(await response.json());
  }
}
