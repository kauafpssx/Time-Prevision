import { displayValue, formatForecastDate, visibleDays } from '../domain/forecast.js';

export class ForecastView {
  constructor(documentRef = document) {
    this.document = documentRef;
    this.showAll = false;
    this.forecast = null;
  }

  bindToggle(handler) {
    this.document.querySelector('#showAll').addEventListener('click', handler);
  }

  toggleDays() {
    this.showAll = !this.showAll;
    this.renderDaily();
  }

  render(forecast) {
    this.forecast = forecast;
    this.renderHero(forecast.daily[0]);
    this.renderHourly(forecast.hourly);
    this.renderDaily();
    const generatedAt = new Date(forecast.generatedAt);
    this.document.querySelector('#updated').textContent =
      `Atualizado em ${generatedAt.toLocaleDateString('pt-BR')} às ${generatedAt.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' })}`;
  }

  renderHero(day) {
    this.document.querySelector('#hero').innerHTML = `<div class="hero-copy"><p>${formatForecastDate(day.date, { weekday: 'long', day: 'numeric', month: 'long' })}</p><h2>${displayValue(Math.round(day.temperature))}°</h2><strong>${day.icon} ${day.condition}</strong><div class="metrics"><div class="metric"><span>MÍN / MÁX</span><b>${displayValue(day.temperatureMin)}° / ${displayValue(day.temperatureMax)}°</b></div><div class="metric"><span>CHUVA</span><b>${displayValue(day.rainChance)}%</b></div><div class="metric"><span>VENTO</span><b>${displayValue(day.windSpeed)} m/s</b></div><div class="metric"><span>UMIDADE</span><b>${displayValue(day.humidity)}%</b></div></div></div>`;
  }

  renderHourly(hours) {
    this.document.querySelector('#hourly').innerHTML = hours.map(hour => `<article class="hour-card"><b>${hour.time}</b><div class="icon">${hour.icon}</div><strong>${displayValue(hour.temperature)}°</strong><br><small>${displayValue(hour.rainChance)}% chuva</small></article>`).join('');
  }

  renderDaily() {
    const days = visibleDays(this.forecast, this.showAll);
    this.document.querySelector('#daily').innerHTML = days.map((day, index) => `<article class="day"><div><b>${index === 0 ? 'Hoje' : formatForecastDate(day.date, { weekday: 'long' })}</b><br><small>${formatForecastDate(day.date, { day: '2-digit', month: 'short' })}</small></div><div class="condition"><span class="icon">${day.icon}</span><span>${day.condition}</span></div><div class="temps"><b>${displayValue(day.temperatureMax)}°</b><b>${displayValue(day.temperatureMin)}°</b></div><div>💧 ${displayValue(day.rainChance)}% <small>· ${displayValue(day.precipitation)} mm</small></div></article>`).join('');
    this.document.querySelector('#showAll').textContent = this.showAll ? 'Ver 7 dias' : 'Ver 30 dias';
  }

  renderError(error) {
    this.document.querySelector('#updated').textContent = error.message;
    this.document.querySelector('#hero').innerHTML = '<div class="hero-copy"><h2>Ops</h2><strong>A previsão ainda não foi publicada.</strong></div>';
  }
}
