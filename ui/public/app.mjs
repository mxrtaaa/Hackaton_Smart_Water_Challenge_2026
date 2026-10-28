
import { calculateCostScenarios, validatePrices } from './economics.mjs';
import { summarizeDetail } from './business-metrics.mjs';
import {
  historyDays,
  selectHistory,
  selectForecast,
  lastMeasuredHour
} from './view-data.mjs';

const $ = s => document.querySelector(s);

const num = (v, d = 0) =>
  v === null || v === undefined
    ? 'brak danych'
    : new Intl.NumberFormat('pl-PL', {
        maximumFractionDigits: d,
        minimumFractionDigits: d
      }).format(v);

const money = v => `${num(v)} zł`;

const esc = v =>
  String(v).replace(
    /[&<>"']/g,
    c => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    })[c]
  );

const date = t =>
  new Intl.DateTimeFormat('pl-PL', {
    timeZone: 'Europe/Warsaw',
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  }).format(new Date(t));

const day = t =>
  new Intl.DateTimeFormat('pl-PL', {
    timeZone: 'Europe/Warsaw',
    day: '2-digit',
    month: '2-digit'
  }).format(new Date(t));

const localDay = t =>
  new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Europe/Warsaw',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit'
  }).format(new Date(t));


// ============================================================
// 2. Stan aplikacji i konfiguracja
// ============================================================

let data;
let range = '30';
let loadId = 0;

let calculationId = 0;
let inputTimer;
let dateScopeKey = null;
let resizeTimer;

const costCache = new Map();
const charts = new Map();

const priceKeys = [
  'lowPrice',
  'basePrice',
  'highPrice'
];

const palette = {
  cyan: '#2dbbd3',
  amber: '#edb35f',
  muted: '#6a8290',
  green: '#6ed7ac',
  red: '#f5847f'
};

const isWeteranow = () => data?.station.id === 'ruda-weteranow';

function chart(
  target,
  rows,
  series,
  { unit = '', asOf = null, marks = false } = {}
) {
  const host = $(target);

  if (!rows.length) {
    host.innerHTML = '<div class="empty">Brak danych w wybranym zakresie</div>';
    charts.delete(target);
    return;
  }

  // Match SVG units to CSS pixels so text is not stretched
  // with the viewport.

  const W = Math.max(240, host.clientWidth - 24);
  const H = host.clientHeight || 260;

  const p = {
    l: 54,
    r: 22,
    t: 34,
    b: 34
  };

  const iw = W - p.l - p.r;
  const ih = H - p.t - p.b;

  const ts = rows.map(r => Date.parse(r.time));

  const min = ts[0];
  const max = ts.at(-1);

  const x = t =>
    p.l + (t - min) / (max - min || 1) * iw;

  const vals = rows
    .flatMap(r => series.map(s => r[s.key]))
    .filter(v => typeof v === 'number' && Number.isFinite(v));

  const floor = Math.min(0, ...vals);
  const ceil = Math.max(1, ...vals) * 1.15;

  const y = v =>
    p.t + ih - (v - floor) / (ceil - floor) * ih;

  const future =
    asOf && Date.parse(asOf) < max
      ? Math.max(p.l, x(Date.parse(asOf)))
      : null;

  // Tworzenie SVG

  let svg = `
    <svg
      viewBox="0 0 ${W} ${H}"
      preserveAspectRatio="xMinYMin meet"
      role="img"
      aria-label="${esc(host.getAttribute('aria-label') || 'Wykres')}"
    >
      <title>Wykres interaktywny · ${esc(unit)}</title>
  `;


  if (future !== null) {
    svg += `
      <rect
        x="${future}"
        y="${p.t}"
        width="${W - p.r - future}"
        height="${ih}"
        fill="#12303b"
        opacity=".4"
      />
      <text
        x="${W - p.r}"
        y="17"
        text-anchor="end"
        class="forecast-caption"
      >
        PREDYKCJA · 30 DNI
      </text>
    `;
  }

  // Poziome linie siatki

  for (let i = 0; i <= 4; i++) {
    const v = floor + (ceil - floor) * i / 4;
    const yy = y(v);

    svg += `
      <line
        x1="${p.l}"
        x2="${W - p.r}"
        y1="${yy}"
        y2="${yy}"
        stroke="#253640"
        stroke-dasharray="3 5"
      />
      <text
        x="${p.l - 10}"
        y="${yy + 3}"
        text-anchor="end"
      >
        ${num(v, v < 10 ? 1 : 0)}
      </text>
    `;
  }

  svg += `
    <text x="${p.l}" y="17">
      ${esc(unit)}
    </text>
  `;

  // Linie serii danych

  for (const s of series) {
    let path = '';
    let pen = false;

    for (const r of rows) {
      const v = r[s.key];

      if (typeof v !== 'number' || !Number.isFinite(v)) {
        pen = false;
        continue;
      }

      path += `
        ${pen ? 'L' : 'M'}
        ${x(Date.parse(r.time)).toFixed(2)},
        ${y(v).toFixed(2)}
      `;

      pen = true;
    }

    svg += `
      <path
        d="${path}"
        fill="none"
        stroke="${s.color}"
        stroke-width="${s.width || 1.65}"
        ${s.dash ? 'stroke-dasharray="6 5"' : ''}
        vector-effect="non-scaling-stroke"
        stroke-linejoin="round"
      />
    `;
  }

  // Zaznaczanie anomalii

  if (marks) {
    for (const r of rows) {
      if (
        r.anomaly &&
        typeof r[series[0].key] === 'number'
      ) {
        svg += `
          <circle
            cx="${x(Date.parse(r.time))}"
            cy="${y(r[series[0].key])}"
            r="3.1"
            fill="#f5847f"
            stroke="#0d171d"
            stroke-width="1.2"
          />
        `;
      }
    }
  }

  // Granica prognozy

  if (future !== null) {
    svg += `
      <line
        x1="${future}"
        x2="${future}"
        y1="${p.t}"
        y2="${p.t + ih}"
        stroke="#426573"
        stroke-dasharray="4 5"
      />
    `;
  }

  // Etykiety osi czasu

  for (let i = 0; i <= 5; i++) {
    const t = min + (max - min) * i / 5;

    svg += `
      <text
        x="${x(t)}"
        y="${H - 12}"
        text-anchor="middle"
      >
        ${day(t)}
      </text>
    `;
  }

  svg += `
    <line
      data-cursor
      x1="0"
      x2="0"
      y1="${p.t}"
      y2="${p.t + ih}"
      stroke="#9cb9c8"
      stroke-dasharray="3 3"
      opacity="0"
    />
  </svg>
  `;

  host.innerHTML = svg;

  const info = {
    host,
    rows,
    series,
    x,
    min,
    max,
    W,
    unit,
    asOf,
    marks,
    index: 0
  };

  charts.set(target, info);

  // Tooltip i synchronizacja kursora

  function show(index, clientX, clientY) {
    info.index = Math.max(
      0,
      Math.min(rows.length - 1, index)
    );

    const r = rows[info.index];
    const t = Date.parse(r.time);

    for (const c of charts.values()) {
      const line = c.host.querySelector('[data-cursor]');

      if (!line) {
        continue;
      }

      if (t >= c.min && t <= c.max) {
        line.setAttribute('x1', c.x(t));
        line.setAttribute('x2', c.x(t));
        line.setAttribute('opacity', '1');
      } else {
        line.setAttribute('opacity', '0');
      }
    }

    const tip = $('#tooltip');

    tip.innerHTML = `
      <strong>
        ${date(r.time)}
        ${r.data_kind === 'prediction' ? '· prognoza' : ''}
      </strong>
    ` + series.map(s => `
      <div class="tooltip-row">
        <span style="color:${s.color}">
          ${esc(s.label)}
        </span>
        <span>
          ${num(r[s.key], 1)} ${esc(unit)}
        </span>
      </div>
    `).join('') + (
      r.anomaly
        ? '<span class="red">● Przekroczenie Q97</span>'
        : ''
    );

    tip.hidden = false;

    tip.style.left = `${
      Math.max(
        8,
        Math.min(clientX + 14, window.innerWidth - tip.offsetWidth - 10)
      )
    }px`;

    tip.style.top = `${
      Math.max(
        8,
        Math.min(clientY + 14, window.innerHeight - tip.offsetHeight - 10)
      )
    }px`;
  }

  host.onpointermove = e => {
    const box = host
      .querySelector('svg')
      .getBoundingClientRect();

    const px = (e.clientX - box.left) / box.width * W;
    const t = min + (px - p.l) / iw * (max - min);

    const i = Math.round(
      (t - min) / (max - min || 1) * (rows.length - 1)
    );

    show(i, e.clientX, e.clientY);
  };

  host.onpointerleave = () => {
    $('#tooltip').hidden = true;

    for (const c of charts.values()) {
      c.host
        .querySelector('[data-cursor]')
        ?.setAttribute('opacity', '0');
    }
  };

  host.onblur = host.onpointerleave;

  host.onkeydown = e => {
    if (
      ['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)
    ) {
      e.preventDefault();

      const i =
        e.key === 'Home'
          ? 0
          : e.key === 'End'
            ? rows.length - 1
            : info.index + (e.key === 'ArrowRight' ? 1 : -1);

      const b = host.getBoundingClientRect();

      show(i, b.left + b.width / 2, b.top + 40);
    }
  };
}

function selectedHistory() {
  return selectHistory(data, historyDays(range));
}

function updateKpis() {
  const last = lastMeasuredHour(data);

  $('#flow-label').textContent =
    'Przepompowano · ostatnia pełna godzina';

  $('#energy-label').textContent =
    'Energia · ostatnia pełna godzina';

  $('#flow').innerHTML = `
    ${num(last.pumpingM3, 1)}
    <span class="unit">m³</span>
  `;

  $('#energy').innerHTML = `
    ${num(last.energyKwh, 1)}
    <span class="unit">kWh</span>
  `;

  const interval = `${
    date(last.time)
  } – ${
    date(Date.parse(last.time) + 3600000)
  }`;

  $('#flow-foot').textContent =
    `Pomiar archiwalny · ${date(last.time)}`;

  $('#energy-foot').textContent =
    `Pomiar archiwalny · ${date(last.time)}`;

  $('#flow-foot').title = interval;
  $('#energy-foot').title = interval;

  const f = data.forecastTotals;
  const w = data.weatherForecast;

  $('#forecast-flow').innerHTML = f
    ? `${num(f.pumpingM3)} <span class="unit">m³</span>`
    : 'Brak prognozy';

  $('#forecast-energy').innerHTML = f
    ? `${num(f.energyKwh)} <span class="unit">kWh</span>`
    : 'Brak prognozy';

  const dates = f
    ? `${day(f.start)}–${day(f.end)} · `
    : '';

  $('#forecast-flow-foot').textContent = f
    ? `${dates}predykcja · ${f.pumpingHours}/${f.expectedHours} h`
    : 'Brak danych pogodowych';

  $('#forecast-energy-foot').textContent = f
    ? `${dates}predykcja · ${f.pricedHours}/${f.expectedHours} h`
    : w?.error || 'Oczekiwanie na pogodę';
}

function renderBusiness(rows, price) {
  const business =
    data.businessEngine === 'python'
      ? data.businessByRange[historyDays(range)]
      : summarizeDetail(rows, price);

  const s = business.summary;

  $('#business-price').textContent = `
    Historia ${historyDays(range)} dni · ${num(price, 2)} zł/kWh
  `;

  const metrics = [
    [
      'Koszt pełnych epizodów',
      money(s.koszt_pelnych_epizodow_zl),
      'koszt_pelnych_epizodow_zl'
    ],
    [
      'Pełne epizody',
      num(s.pelne_epizody),
      'pelne_epizody'
    ],
    [
      'Godziny mokre',
      num(s.godziny_mokre),
      'godziny_mokre'
    ],
    [
      'Godziny z wyceną',
      `${num(s.godziny_wycenione)} / ${num(s.godziny_mokre)}`,
      'godziny_wycenione'
    ]
  ];

  $('#business-stats').innerHTML = metrics
    .map(([label, value, key]) => `
      <div class="business-stat">
        <label>${label}</label>
        <strong>${value}</strong>
        <small>${key}</small>
      </div>
    `)
    .join('');

  $('#episodes').innerHTML = business.episodes
    .map(e => `
      <tr>
        <td>${date(e.poczatek)}</td>
        <td>${e.godziny} / ${e.godziny_z_pomiarami}</td>
        <td>${num(e.nadwyzka_m3, 1)}</td>
        <td>${num(e.nadwyzka_kwh, 1)}</td>
        <td>
          ${e.koszt_zl === null
            ? 'brak wyceny'
            : money(e.koszt_zl)}
        </td>
        <td>${e.kompletny ? 'tak' : 'nie'}</td>
      </tr>
    `)
    .join('') ||
    '<tr><td colspan="6">Brak mokrych epizodów.</td></tr>';

  $('#monthly').innerHTML = business.monthly
    .map(m => `
      <tr>
        <td>${m.miesiac}</td>
        <td>${m.godziny_mokre}</td>
        <td>${m.godziny_wycenione}</td>
        <td>${num(m.nadwyzka_m3, 1)}</td>
        <td>${num(m.nadwyzka_kwh, 1)}</td>
        <td>${num(m.pokrycie * 100, 1)}%</td>
        <td>
          ${m.bilans_zl === null
            ? 'brak wyceny'
            : money(m.bilans_zl)}
        </td>
      </tr>
    `)
    .join('');
}

function renderCharts() {
  if (!data) {
    return;
  }

  const future = selectForecast(data);
  const isFuture = range === 'future';

  const rows = isFuture
    ? future
    : selectedHistory();

  $('#flow-subtitle').textContent = isFuture
    ? `PREDYKCJE · ${
        future.length
          ? day(future[0].time) + '–' + day(future.at(-1).time)
          : 'kolejne 30 dni'
      }`
    : `POMIARY · ${range} dni archiwum do ${day(data.asOf)}`;

  $('#forecast-note').textContent = isFuture
    ? data.weatherForecast?.note ||
      data.weatherForecast?.error ||
      'Brak prognozy pogody.'
    : `Historia jest liczona wstecz od ${date(data.asOf)}; nie są to pomiary na żywo.`;

  $('#quantile-note').textContent = isFuture
    ? 'Q97 to górny kwantyl modelu przy zadanej pogodzie. Nie jest drugą prognozą bazową ani pełnym zakresem niepewności na 30 dni.'
    : 'Q97 służy do identyfikacji przekroczeń pomiarów; nie określa przyczyny.';

  document.querySelectorAll('.actual-legend').forEach(el => {
    el.hidden = isFuture;
  });

  document.querySelectorAll('.anomaly-legend').forEach(el => {
    el.hidden = isFuture;
  });

  if (isFuture && !future.length) {
    $('#tooltip').hidden = true;

    for (const id of ['#flow-chart', '#energy-chart']) {
      charts.delete(id);

      const host = $(id);

      host.onpointermove = null;
      host.onpointerleave = null;
      host.onkeydown = null;
      host.onblur = null;

      host.innerHTML = `
        <div class="empty">
          Prognoza niedostępna. Brakujące dane pogodowe
          nie są zastępowane pomiarami ani zerami.
        </div>
      `;
    }
  } else {
    const flow = [
      {
        key: 'prognoza',
        label: 'Predykcja bazowa',
        color: palette.amber,
        dash: true,
        width: 2
      },
      {
        key: 'q97',
        label: 'Q97 · górny kwantyl',
        color: palette.muted,
        dash: true,
        width: 1
      }
    ];

    const energy = [
      {
        key: 'energia_prognoza',
        label: isFuture ? 'Predykcja' : 'Przewidywane',
        color: palette.amber,
        dash: true
      }
    ];

    if (isWeteranow()) {
      energy.push({
        key: 'energia_q97',
        label: 'Q97 energii',
        color: palette.muted,
        dash: true,
        width: 1
      });
    }

    if (!isFuture) {
      flow.unshift({
        key: 'przepompowano_m3',
        label: 'Pomiar',
        color: palette.cyan
      });

      energy.unshift({
        key: 'energia_kwh',
        label: 'Pomiar',
        color: palette.cyan
      });
    }

    chart('#flow-chart', rows, flow, {
      unit: 'm³/h',
      asOf: isFuture ? rows[0].time : null,
      marks: !isFuture
    });

    chart('#energy-chart', rows, energy, {
      unit: 'kWh / godz.',
      asOf: isFuture ? rows[0].time : null
    });
  }

  updateKpis();

  renderBusiness(
    selectedHistory(),
    data.pricePerKwh
  );

  calculate();
}

function renderAnomalies() {
  $('#anomaly-count').textContent = data.anomalies.length;
  const costCell = (value, valuation) => {
    const coverage = valuation
      ? `${valuation.godziny_wycenione}/${valuation.godziny} godzin z wyceną`
      : 'Brak danych wyceny z backendu Python';
    return `<td title="${esc(coverage)}">${Number.isFinite(value)
      ? `${num(value, 2)} zł`
      : 'brak wyceny'}<small>${esc(coverage)}</small></td>`;
  };
  $('#anomaly-cost-note').textContent = data.anomalyCostMethod
    ? `Wycena: ${num(data.anomalyCostMethod.pricePerKwh, 2)} zł/kWh z business_metrics.py. `
      + 'Pogoda: model energii z opadami i przewidywanym pompowaniem. '
      + 'Bez pogody: mediany wszystkich kompletnych godzin według miesiąca, godziny i weekendu '
      + '(minimum 5 próbek; fallback jak w business_metrics.py). '
      + 'Koszt = dodatnia część sumy różnic kWh × cena, tylko za godziny przekroczenia Q97. '
      + 'Kolumny pokazują alternatywne wyceny tej samej nadwyżki. '
      + 'Brak wymaganej godziny oznacza brak pełnej wyceny. To szacunek nadwyżki kosztu energii.'
    : 'Koszty wymagają wyceny przez backend Python; nie są zastępowane zerami.';

  $('#anomalies').innerHTML = data.anomalies
    .map(a => `
      <tr>
        <td>
          <strong>${esc(a.title)}</strong>
          <small>${date(a.time)} – ${date(a.end)}</small>
        </td>
        <td class="amber">
          ${a.deviation === null
            ? 'brak danych'
            : num(a.deviation, 1) + '%'}
        </td>
        <td>${num(a.ponad_q97, 1)}</td>
        ${costCell(a.koszt_pogodowy_zl, a.wycena_pogodowa)}
        ${costCell(a.koszt_kalendarzowy_zl, a.wycena_kalendarzowa)}
      </tr>
    `)
    .join('') ||
    '<tr><td colspan="5" class="empty">Brak zarejestrowanych przekroczeń Q97.</td></tr>';
}

function sourcePrices() {
  for (const key of priceKeys) {
    $(`[name="${key}"]`).value = data.pricePerKwh;
  }
}

function calculationScope() {
  const choice = $('#calc-period').value;

  if (choice === 'view') {
    return {
      period: range === 'future' ? 'forecast' : 'history',
      days: historyDays(range)
    };
  }

  return {
    period: choice === 'forecast30' ? 'forecast' : 'history',
    days: Number(choice.replace(/\D/g, ''))
  };
}

function customCostOptions(scope) {
  $('#calc-custom-dates').hidden = !isWeteranow();

  if (!isWeteranow()) {
    return {};
  }

  const rows = scope.period === 'forecast'
    ? data.forecast
    : selectHistory(data, scope.days);

  if (!rows.length) {
    throw new Error('Brak danych dla wybranego okresu.');
  }

  const first = localDay(rows[0].time);
  const last = localDay(rows.at(-1).time);

  const key = `
    ${data.station.id}|${scope.period}|${scope.days}
  `.trim();

  if (dateScopeKey !== key) {
    $('#calc-start').value = first;
    $('#calc-end').value = last;

    dateScopeKey = key;

    for (const id of ['#calc-start', '#calc-end']) {
      $(id).min = first;
      $(id).max = last;
    }
  }

  $('#calc-basis').disabled = scope.period === 'forecast';

  return {
    startDate: $('#calc-start').value,
    endDate: $('#calc-end').value,
    energyBasis: scope.period === 'forecast'
      ? 'predicted'
      : $('#calc-basis').value
  };
}

async function calculate() {
  if (!data) {
    return;
  }

  const token = ++calculationId;
  const scope = calculationScope();
  const metric = $('#calc-metric').value;

  $('#calc-results').hidden = true;
  $('#calc-error').hidden = true;

  const type = scope.period === 'forecast'
    ? 'PREDYKCJA · przyszłe 30 dni'
    : `${
        isWeteranow() && $('#calc-basis').value === 'predicted'
          ? 'PREDYKCJA HISTORYCZNA'
          : 'POMIAR'
      } · ${scope.days} dni historii`;

  $('#calc-source').textContent = type + ' · obliczanie…';

  try {
    // Odczyt cen operatora

    const prices = Object.fromEntries(
      priceKeys.map(key => {
        const v = $(`[name="${key}"]`).value;

        return [
          key,
          v.trim() === '' ? NaN : Number(v)
        ];
      })
    );

    validatePrices(prices);

    let scenarios;

    if (scope.period === 'forecast' && !data.forecast.length) {
      throw new Error(
        data.weatherForecast?.error ||
        'Brak predykcji dla przyszłych dat. Wycena przyszłości nie może używać pomiarów historycznych.'
      );
    }

    // Kalkulator oparty na backendzie Python

    if (data.businessEngine === 'python') {
      const input = {
        station: data.station.id,
        ...scope,
        metric,
        prices,
        ...customCostOptions(scope)
      };

      const key = JSON.stringify(input);

      scenarios = costCache.get(key);

      if (!scenarios) {
        const response = await fetch('/api/cost-scenarios', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: key
        });

        const body = await response.json();

        if (!response.ok) {
          throw new Error(body.error || 'Błąd wyceny.');
        }

        scenarios = body;
        costCache.set(key, scenarios);
      }
    } else {
      // Kalkulator lokalny

      if (isWeteranow()) {
        throw new Error(
          'Kalkulator Weteranów wymaga połączonego backendu Python.'
        );
      }

      if (
        scope.period !== 'history' ||
        metric !== 'wet_excess'
      ) {
        throw new Error(
          'Ta opcja wymaga połączonego backendu Python.'
        );
      }

      scenarios = calculateCostScenarios(
        selectHistory(data, scope.days),
        prices
      );
    }

    if (token !== calculationId) {
      return;
    }

    $('#calc-source').textContent = type;
    $('#calc-results').hidden = false;

    // Funkcja formatowania kosztu

    const cost = x =>
      x.valuation
        ? x.valuation.totalCostZl === null
          ? 'brak danych'
          : money(x.valuation.totalCostZl)
        : money(x.summary.koszt_pelnych_epizodow_zl);

    const b = scenarios[1];
    const v = b.valuation;

    const expected = scope.period === 'forecast'
      ? 'Prognozowany koszt'
      : 'Koszt pomiarów';

    // Podstawowe wyniki kalkulatora

    $('#calc-results').innerHTML = `
      <article class="decision">
        <div class="decision-top">
          <span class="step">${type}</span>
          <span class="badge">
            ${num(prices.basePrice, 2)} zł/kWh
          </span>
        </div>

        <h3>${cost(b)}</h3>

        <p>
          ${expected} · ${
            metric === 'energy'
              ? 'całe zużycie energii'
              : 'nadwyżki w kompletnych mokrych epizodach'
          }.
        </p>

        <p>
          Zakres cenowy:
          ${cost(scenarios[0])} – ${cost(scenarios[2])}.
        </p>

        <p class="result-note">
          ${
            v
              ? `${v.pricedHours}/${v.expectedHours} godzin z wyceną · ${
                  v.complete
                    ? 'pełne pokrycie'
                    : 'UWAGA: wycena częściowa; nie obejmuje brakujących godzin'
                }`
              : ''
          }.
          ${
            scope.period === 'forecast'
              ? 'Podstawą są wyłącznie predykcje modeli, nie rzeczywiste pomiary.'
              : 'Podstawą są archiwalne pomiary.'
          }
        </p>
      </article>

      <div class="scenario-card">
        <h3>Zakres szacunku według ceny kWh</h3>

        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Wariant</th>
                ${
                  scenarios
                    .map(x => `<th>${x.name}</th>`)
                    .join('')
                }
              </tr>
            </thead>

            <tbody>
              <tr>
                <th>Cena zł/kWh</th>
                ${
                  scenarios
                    .map(x => `<td>${num(x.price, 2)}</td>`)
                    .join('')
                }
              </tr>

              <tr>
                <th>
                  ${
                    metric === 'energy'
                      ? 'Koszt energii'
                      : 'Koszt mokrych epizodów'
                  }
                </th>
                ${
                  scenarios
                    .map(x => `<td>${cost(x)}</td>`)
                    .join('')
                }
              </tr>
            </tbody>
          </table>
        </div>

        <p>
          To zakres cen wpisanych przez operatora,
          a nie przedział ufności modelu ani obietnica oszczędności.
        </p>
      </div>
    `;

    // Dodatkowe szczegóły dla Weteranów

    if (isWeteranow() && v) {
      const card = $('#calc-results .decision');

      card.querySelector('h3').insertAdjacentHTML(
        'afterend',
        `
          <p>
            Suma: ${num(v.energyKwh, 2)} kWh ·
            ${num(v.pumpingM3, 2)} m³
            <br>
            ${date(v.start)} – ${date(v.end)}
          </p>
        `
      );

      if (
        v.dataKind === 'prediction' &&
        scope.period === 'history'
      ) {
        card.querySelector('.step').textContent =
          'PREDYKCJA HISTORYCZNA · WETERANÓW';

        card.querySelector('h3 + p + p').textContent =
          'Koszt predykcji historycznej · ' +
          (
            metric === 'energy'
              ? 'całe zużycie energii'
              : 'nadwyżki mokrych epizodów'
          );

        card.querySelector('.result-note').textContent = `
          ${v.pricedHours}/${v.expectedHours}
          godzin z wyceną ·
          ${v.complete ? 'pełne pokrycie' : 'wycena częściowa'}.
          Podstawą są predykcje modeli dla dat historycznych.
        `;
      }
    }

    // Dzienna wycena energii

    if (metric === 'energy' && v?.daily?.length) {
      $('#calc-results').innerHTML += `
        <details class="card methodology">
          <summary>
            Wycena dzień po dniu · ${
              v?.dataKind === 'prediction'
                ? 'predykcje'
                : 'pomiary'
            }
          </summary>

          <div class="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Dzień</th>
                  <th>Energia kWh</th>
                  <th>Koszt zł</th>
                  <th>Godziny / dostępne</th>
                </tr>
              </thead>

              <tbody>
                ${
                  v.daily.map(d => `
                    <tr>
                      <td>${d.date}</td>
                      <td>${num(d.energyKwh, 1)}</td>
                      <td>
                        ${
                          d.costZl === null
                            ? 'brak danych'
                            : money(d.costZl)
                        }
                      </td>
                      <td>${d.pricedHours}/${d.expectedHours}</td>
                    </tr>
                  `).join('')
                }
              </tbody>
            </table>
          </div>
        </details>
      `;
    }

  } catch (error) {
    if (token !== calculationId) {
      return;
    }

    $('#calc-error').textContent = error.message;
    $('#calc-error').hidden = false;
    $('#calc-results').hidden = true;
  }
}

async function load() {
  const token = ++loadId;

  ++calculationId;

  clearTimeout(inputTimer);
  costCache.clear();

  dateScopeKey = null;

  $('#dashboard').setAttribute('aria-busy', 'true');
  $('#refresh').disabled = true;
  $('#load-error').hidden = true;

  try {
    const station = $('#station').value;

    $('#dashboard').hidden = true;

    $('#station-code').textContent =
      station === 'ruda-weteranow'
        ? 'RS-01'
        : 'CH-01';

    $('#as-of').textContent =
      'Wczytywanie danych wybranej lokalizacji…';

    $('#data-notice').textContent =
      'Wczytywanie właściwych danych i modeli. Poprzednia lokalizacja nie jest używana jako zastępstwo.';

    history.replaceState(
      null,
      '',
      `?station=${encodeURIComponent(station)}${location.hash}`
    );

    // Aktywna lokalizacja w nawigacji

    document.querySelectorAll('[data-station]').forEach(link => {
      const active = link.dataset.station === station;

      link.classList.toggle('selected', active);

      if (active) {
        link.setAttribute('aria-current', 'page');
      } else {
        link.removeAttribute('aria-current');
      }
    });

    // Pobranie danych z API

    const response = await fetch(
      '/api/dashboard?station=' + encodeURIComponent(station)
    );

    const body = await response.json();

    if (!response.ok) {
      throw new Error(body.error || 'Błąd źródła danych.');
    }

    if (token !== loadId) {
      return;
    }

    data = body;

    // Walidacja kontraktu danych

    if (
      data.schemaVersion !== 2 ||
      !data.history?.length ||
      !Array.isArray(data.forecast) ||
      !Number.isFinite(data.pricePerKwh)
    ) {
      throw new Error('Nieobsługiwany kontrakt danych.');
    }

    selectForecast(data);

    $('#station').value = data.station.id;

    $('.demo-tag').textContent =
      data.source === 'demo'
        ? 'DANE PRZYKŁADOWE'
        : 'POMIARY / PREDYKCJE';

    $('#data-notice').textContent =
      data.source === 'demo'
        ? 'Tryb demonstracyjny.'
        : `
          ${data.station.name}: pomiary do ${date(data.asOf)}
          (ostatnia pełna godzina archiwum), bez uzupełniania braków.
          Historia 7/14/30 dni jest liczona wstecz od tej daty.
          Prognoza to oddzielne przyszłe daty i wyłącznie wyniki
          modeli z pogodą Open-Meteo.
          ${
            data.weatherForecast?.status === 'unavailable'
              ? 'Prognoza obecnie niedostępna: ' +
                data.weatherForecast.error
              : ''
          }
        `;

    $('#energy-description').textContent = isWeteranow()
      ? 'Weteranów: predykcja energii z przewidywanego pompowania i pogody; pomiar energii pokazany osobno.'
      : 'Wartości godzinowe · ta sama oś czasu co przepompowanie';

    $('#energy-q97-legend').hidden = !isWeteranow();
    $('#calc-custom-dates').hidden = !isWeteranow();

    $('#footer-location').textContent = data.station.name;

    $('#flow-model').textContent = 'CatBoost · Q97';

    $('#station-code').textContent = data.station.code;

    $('#as-of').textContent = `
      Koniec historii: ${date(data.asOf)} · Europe/Warsaw
    `;

    sourcePrices();

    $('#dashboard').hidden = false;

    renderCharts();
    renderAnomalies();

  } catch (error) {
    if (token !== loadId) {
      return;
    }

    $('#dashboard').hidden = true;

    $('#load-error').textContent =
      `Nie udało się wczytać danych: ${error.message}`;

    $('#load-error').hidden = false;

  } finally {
    if (token === loadId) {
      $('#dashboard').setAttribute('aria-busy', 'false');
      $('#refresh').disabled = false;
    }
  }
}


// Zmiana przepompowni

$('#station').addEventListener('change', load);

// Odświeżanie danych

$('#refresh').addEventListener('click', load);

// Zmiana zakresu czasu

$('#range').addEventListener('click', e => {
  const b = e.target.closest('[data-range]');

  if (!b) {
    return;
  }

  range = b.dataset.range;

  $('#range .selected')?.classList.remove('selected');
  b.classList.add('selected');

  renderCharts();
});

// Formularz kalkulatora

$('#calculator-form').addEventListener('submit', e => {
  e.preventDefault();
});

$('#calculator-form').addEventListener('input', e => {
  ++calculationId;

  $('#calc-results').hidden = true;

  clearTimeout(inputTimer);

  if (e.target.id === 'calc-period') {
    dateScopeKey = null;
  }

  inputTimer = setTimeout(calculate, 180);
});

// Reset cen

$('#reset').addEventListener('click', () => {
  if (!data) {
    return;
  }

  sourcePrices();
  calculate();
});

// Eksport danych dashboardu do JSON

$('#download-data').addEventListener('click', e => {
  e.preventDefault();

  if (!data) {
    return;
  }

  const url = URL.createObjectURL(
    new Blob(
      [
        JSON.stringify(
          { dashboards: [data] },
          null,
          2
        )
      ],
      {
        type: 'application/json'
      }
    )
  );

  const a = document.createElement('a');

  a.href = url;
  a.download = `smart-water-${data.station.id}.json`;

  a.click();

  URL.revokeObjectURL(url);
});

// Dopasowanie wykresów do rozmiaru okna

window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);

  resizeTimer = setTimeout(() => {
    for (const [target, c] of [...charts]) {
      chart(target, c.rows, c.series, {
        unit: c.unit,
        asOf: c.asOf,
        marks: c.marks
      });
    }
  }, 100);
});

const initialStation = new URLSearchParams(
  location.search
).get('station');

if (['chorzow', 'ruda-weteranow'].includes(initialStation)) {
  $('#station').value = initialStation;
}

load();
