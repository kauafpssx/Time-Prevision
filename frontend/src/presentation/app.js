import { LoadForecast } from '../application/loadForecast.js';
import { ForecastRepository } from '../infrastructure/forecastRepository.js';
import { ForecastView } from './forecastView.js';

const view = new ForecastView();
const loadForecast = new LoadForecast(new ForecastRepository());

view.bindToggle(() => view.toggleDays());

try {
  view.render(await loadForecast.execute());
} catch (error) {
  view.renderError(error);
}
