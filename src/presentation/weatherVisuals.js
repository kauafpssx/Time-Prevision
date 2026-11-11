export function weatherKind(item = {}) {
  const condition = String(item.condition || '').toLowerCase();
  if (condition.includes('tempest') || condition.includes('temporal')) return 'storm';
  if (condition.includes('chuva forte') || condition.includes('chuva moderada')) return 'rain';
  if (condition.includes('chuva') || condition.includes('garoa')) return 'drizzle';
  if (Number(item.cloudiness) >= 6) return 'cloudy';
  if (Number(item.cloudiness) >= 3) return 'partly';
  return 'clear';
}

export function weatherIcon(item, className = '') {
  const kind = weatherKind(item);
  const sun = '<circle class="wi-sun" cx="24" cy="24" r="8"/><g class="wi-rays"><path d="M24 5v5M24 38v5M5 24h5M38 24h5M10.6 10.6l3.6 3.6M33.8 33.8l3.6 3.6M37.4 10.6l-3.6 3.6M14.2 33.8l-3.6 3.6"/></g>';
  const cloud = '<path class="wi-cloud" d="M13 34h23a7 7 0 0 0 .6-14A12 12 0 0 0 14 22.5 6 6 0 0 0 13 34Z"/>';
  const rain = '<g class="wi-rain"><path d="m17 38-2 5M26 38l-2 5M35 38l-2 5"/></g>';
  const drizzle = '<g class="wi-rain"><path d="m20 39-1 3M31 39l-1 3"/></g>';
  const lightning = '<path class="wi-lightning" d="m27 35-5 9 6-2-2 7 9-12-6 2 3-4Z"/>';
  let content = sun;
  if (kind === 'partly') content = `<g transform="translate(-5 -5) scale(.78)">${sun}</g>${cloud}`;
  if (kind === 'cloudy') content = cloud;
  if (kind === 'drizzle') content = `${cloud}${drizzle}`;
  if (kind === 'rain') content = `${cloud}${rain}`;
  if (kind === 'storm') content = `${cloud}${lightning}`;
  return `<svg class="weather-svg ${className}" data-kind="${kind}" viewBox="0 0 48 52" aria-hidden="true">${content}</svg>`;
}

export function atmosphere(kind) {
  const clouds = '<i class="scene-cloud cloud-a"></i><i class="scene-cloud cloud-b"></i><i class="scene-cloud cloud-c"></i>';
  if (kind === 'clear') return '<div class="scene-sun"><i></i></div><div class="scene-haze"></div>';
  if (kind === 'partly' || kind === 'cloudy') return `<div class="scene-sun muted"><i></i></div>${clouds}`;
  const drops = Array.from({ length: kind === 'drizzle' ? 22 : 42 }, (_, index) => `<i style="--x:${(index * 37) % 101}%;--delay:-${(index % 13) * .17}s;--speed:${.72 + (index % 7) * .08}s;--length:${12 + (index % 5) * 5}px"></i>`).join('');
  const flash = kind === 'storm' ? '<div class="scene-lightning"></div>' : '';
  return `${clouds}<div class="scene-rain">${drops}</div>${flash}`;
}
