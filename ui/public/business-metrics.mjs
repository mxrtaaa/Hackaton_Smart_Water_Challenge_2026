
const HOUR = 3600000;

const valid = x =>
  typeof x === 'number' && Number.isFinite(x);

const sum = a =>
  a.some(valid)
    ? a.filter(valid).reduce((s, x) => s + x, 0)
    : null;

const median = a => {
  const s = [...a].sort((a, b) => a - b);

  return (
    s[Math.floor((s.length - 1) / 2)] +
    s[Math.floor(s.length / 2)]
  ) / 2;
};

const local = new Intl.DateTimeFormat('en-CA', {
  timeZone: 'Europe/Warsaw',
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  hourCycle: 'h23',
  weekday: 'short'
});


function calendar(time) {
  const p = Object.fromEntries(
    local.formatToParts(new Date(time))
      .map(p => [p.type, p.value])
  );

  return {
    miesiac: Number(p.month),
    godzina: Number(p.hour),
    weekend: ['Sat', 'Sun'].includes(p.weekday) ? 1 : 0,
    monthKey: `${p.year}-${p.month}`
  };
}


export function classifyWeatherPeriods(input, dry = 0.2, wet = 1) {
  if (!input.length) {
    return [];
  }

  const byTime = new Map();

  for (const r of input) {
    if (
      !/(Z|[+-]\d\d:\d\d)$/.test(r.time) ||
      !Number.isFinite(Date.parse(r.time))
    ) {
      throw new Error('Czas musi mieć strefę.');
    }

    const t = Date.parse(r.time);

    if (byTime.has(t)) {
      throw new Error('Duplikaty dat.');
    }

    if (valid(r.opad_mm) && r.opad_mm < 0) {
      throw new Error('Ujemne opady.');
    }

    byTime.set(t, r);
  }

  const times = [...byTime.keys()].sort((a, b) => a - b);
  const out = [];

  for (let t = times[0]; t <= times.at(-1); t += HOUR) {
    const r = {
      przepompowano_m3: null,
      energia_kwh: null,
      opad_mm: null,
      ...byTime.get(t),
      time: new Date(t).toISOString(),
      ...calendar(t)
    };

    out.push(r);

    for (const h of [24, 48]) {
      const a = out.slice(-h).map(r => r.opad_mm);

      r[`opad_${h}h`] =
        a.length === h && a.every(valid)
          ? sum(a)
          : null;
    }

    r.okres =
      valid(r.opad_24h) && valid(r.opad_48h)
        ? r.opad_48h < dry
          ? 'suchy'
          : r.opad_24h >= wet
            ? 'mokry'
            : 'przejsciowy'
        : 'nieokreslony';
  }

  return out;
}


export function calculateDryBaseline(rows, minSamples = 5) {
  const dry = rows.filter(
    r =>
      r.okres === 'suchy' &&
      valid(r.przepompowano_m3) &&
      valid(r.energia_kwh)
  );

  if (dry.length < minSamples) {
    throw new Error('Za mało kompletnych godzin suchych.');
  }

  const levels = [
    [
      'miesiac+godzina+weekend',
      ['miesiac', 'godzina', 'weekend']
    ],
    [
      'miesiac+godzina',
      ['miesiac', 'godzina']
    ],
    [
      'godzina+weekend',
      ['godzina', 'weekend']
    ],
    [
      'godzina',
      ['godzina']
    ],
    [
      'miesiac',
      ['miesiac']
    ]
  ];

  const stats = levels.map(([label, keys]) => {
    const groups = new Map();

    for (const r of dry) {
      const k = keys.map(k => r[k]).join('|');

      if (!groups.has(k)) {
        groups.set(k, []);
      }

      groups.get(k).push(r);
    }

    return {
      label,
      keys,
      groups: new Map(
        [...groups].map(([k, a]) => [
          k,
          {
            n: a.length,
            m3: median(a.map(r => r.przepompowano_m3)),
            kwh: median(a.map(r => r.energia_kwh))
          }
        ])
      )
    };
  });

  return rows.map(r => {
    const out = {
      ...r,
      bazowe_m3: null,
      bazowe_kwh: null,
      baseline_zrodlo: null,
      baseline_n: null
    };

    for (const l of stats) {
      const s = l.groups.get(
        l.keys.map(k => r[k]).join('|')
      );

      if (s && s.n >= minSamples) {
        Object.assign(out, {
          bazowe_m3: s.m3,
          bazowe_kwh: s.kwh,
          baseline_zrodlo: l.label,
          baseline_n: s.n
        });

        break;
      }
    }

    return out;
  });
}


export function calculateExcess(rows, price = 1) {
  return rows.map(r => {
    const ok =
      r.okres === 'mokry' &&
      [
        r.przepompowano_m3,
        r.energia_kwh,
        r.bazowe_m3,
        r.bazowe_kwh
      ].every(valid);

    const m3 = ok
      ? r.przepompowano_m3 - r.bazowe_m3
      : null;

    const kwh = ok
      ? r.energia_kwh - r.bazowe_kwh
      : null;

    return {
      ...r,
      nadwyzka_m3: m3,
      nadwyzka_kwh: kwh,
      koszt_nadwyzki_zl: valid(kwh)
        ? kwh * price
        : null
    };
  });
}


export function summarizeDetail(rows, price = 1) {
  const episodes = [];

  let current = null;
  let id = 0;

  for (const r of rows) {
    if (r.okres !== 'mokry') {
      current = null;
      continue;
    }

    if (!current) {
      current = {
        epizod_id: ++id,
        poczatek: r.time,
        koniec: r.time,
        rows: []
      };

      episodes.push(current);
    }

    current.koniec = r.time;
    current.rows.push(r);
  }

  const result = episodes
    .map(e => {
      const n = e.rows.filter(
        r => valid(r.nadwyzka_kwh)
      ).length;

      const complete = n === e.rows.length;

      const kwh = complete
        ? sum(e.rows.map(r => r.nadwyzka_kwh))
        : null;

      return {
        epizod_id: e.epizod_id,
        poczatek: e.poczatek,
        koniec: e.koniec,
        godziny: e.rows.length,
        godziny_z_pomiarami: n,
        kompletny: complete,
        nadwyzka_m3: complete
          ? sum(e.rows.map(r => r.nadwyzka_m3))
          : null,
        nadwyzka_kwh: kwh,
        koszt_zl: valid(kwh)
          ? Math.max(0, kwh) * price
          : null
      };
    })
    .sort(
      (a, b) =>
        (b.koszt_zl ?? -Infinity) -
        (a.koszt_zl ?? -Infinity)
    );

  const months = new Map();

  for (const r of rows.filter(r => r.okres === 'mokry')) {
    const key = r.monthKey ?? calendar(r.time).monthKey;

    if (!months.has(key)) {
      months.set(key, []);
    }

    months.get(key).push(r);
  }

  const monthly = [...months].map(([miesiac, a]) => {
    const n = a.filter(
      r => valid(r.nadwyzka_kwh)
    ).length;

    const kwh = sum(
      a.map(r => r.nadwyzka_kwh)
    );

    return {
      miesiac,
      godziny_mokre: a.length,
      godziny_wycenione: n,
      nadwyzka_m3: sum(
        a.map(r => r.nadwyzka_m3)
      ),
      nadwyzka_kwh: kwh,
      pokrycie: n / a.length,
      bilans_zl: kwh === null
        ? null
        : kwh * price
    };
  });

  return {
    episodes: result,
    monthly,
    summary: {
      godziny_mokre: rows.filter(
        r => r.okres === 'mokry'
      ).length,

      godziny_wycenione: rows.filter(
        r =>
          r.okres === 'mokry' &&
          valid(r.nadwyzka_kwh)
      ).length,

      pelne_epizody: result.filter(
        r => r.kompletny
      ).length,

      koszt_pelnych_epizodow_zl: result.reduce(
        (s, r) => s + (r.koszt_zl ?? 0),
        0
      )
    }
  };
}


export function runBusinessAnalysis(
  rows,
  {
    price = 1,
    dry = 0.2,
    wet = 1,
    minSamples = 5
  } = {}
) {
  const detail = calculateExcess(
    calculateDryBaseline(
      classifyWeatherPeriods(rows, dry, wet),
      minSamples
    ),
    price
  );

  return {
    detail,
    ...summarizeDetail(detail, price)
  };
}
