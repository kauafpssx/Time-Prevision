import { LoadForecast } from '../application/loadForecast.js';
import { ForecastRepository } from '../infrastructure/forecastRepository.js';
import { ForecastView } from './forecastView.js';

const view=new ForecastView();
const loadForecast=new LoadForecast(new ForecastRepository());
view.bindRange(range=>view.setRange(range));
view.bindTheme(()=>{document.body.classList.toggle('light');localStorage.setItem('atmos-theme',document.body.classList.contains('light')?'light':'dark')});
if(localStorage.getItem('atmos-theme')==='light')document.body.classList.add('light');
try{view.render(await loadForecast.execute())}catch(error){view.renderError(error)}
