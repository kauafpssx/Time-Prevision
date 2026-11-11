import { LoadForecast } from '../application/loadForecast.js?v=20261111-3';
import { ForecastRepository } from '../infrastructure/forecastRepository.js?v=20261111-3';
import { ForecastView } from './forecastView.js?v=20261111-3';

const view=new ForecastView();
const loadForecast=new LoadForecast(new ForecastRepository());
view.bindRange(range=>view.setRange(range));
view.bindTheme(()=>{document.body.classList.toggle('light');localStorage.setItem('atmos-theme',document.body.classList.contains('light')?'light':'dark')});
if(localStorage.getItem('atmos-theme')==='light')document.body.classList.add('light');
try{view.render(await loadForecast.execute())}catch(error){view.renderError(error)}
