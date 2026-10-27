import { runBusinessAnalysis } from './public/business-metrics.mjs';

// Wyłącznie szeregi demonstracyjne UI, nie pomiary ani inferencja modeli.
export const stations = [
    {
        id: 'chorzow',
        name: 'Chorzów',
        code: 'CH-01',
        base: 155,
        energy: 30
    },
    {
        id: 'ruda-weteranow',
        name: 'Ruda Śląska · Weteranów',
        code: 'RS-01',
        base: 42,
        energy: 16
    }
];

const H = 3600000;
const anchor = Date.parse('2026-10-27T12:00:00+01:00');
const round = x => Math.round(x * 100) / 100;

export function getDashboard(id = 'chorzow') {
    const station = stations.find(s => s.id === id);

    if (!station) {
        throw new Error('Nieznany obiekt.');
    }

    const history = [];

    for (let i = -120 * 24; i <= 0; i++) {
        const t = anchor + i * H;
        const day = Math.floor(i / 24);
        const hour = ((i + 12) % 24 + 24) % 24;
        const phase = ((day % 13) + 13) % 13;

        const rain = phase === 3 || phase === 4
            ? Math.max(0, Math.sin(hour / 3)) * .85
            : 0;

        const wet = phase >= 3 && phase <= 5;

        const base = station.base * (
            1
            + .13 * Math.sin((hour - 6) / 24 * Math.PI * 2)
            + .035 * Math.cos(day / 7 * Math.PI * 2)
        );

        const predicted = base
            + (wet ? station.base * .20 : 0)
            + rain * station.base * .15;

        const event =
            (i >= -16 && i <= 0) ||
            (i >= -8 * 24 && i < -8 * 24 + 14) ||
            (i >= -19 * 24 && i < -19 * 24 + 9);

        const actual = predicted * (1 + .025 * Math.sin(i * 1.7))
            + (event ? station.base * .43 : 0);

        const energyPred = station.energy * (predicted / station.base) ** .95;

        const r = {
            time: new Date(t).toISOString(),
            przepompowano_m3: round(actual),
            energia_kwh: round(
                energyPred + (event ? 12.5 : 0) + Math.sin(i * 1.3) * .7
            ),
            opad_mm: round(rain),
            prognoza: round(predicted),
            q97: round(predicted * 1.22),
            energia_prognoza: round(energyPred),
            energia_q97: round(energyPred * 1.20),
            anomaly: actual > predicted * 1.22
        };

        history.push(r);
    }

    // Pokazujemy również niekompletny epizod: brak pomiaru nigdy nie staje się zerem.
    const missing = history.find(
        r => r.time === new Date(anchor - 8 * 24 * H + 3 * H).toISOString()
    );

    if (missing) {
        missing.energia_kwh = null;
        missing.przepompowano_m3 = null;
        missing.anomaly = false;
    }

    const business = runBusinessAnalysis(history);

    const visible = business.detail.filter(
        r => Date.parse(r.time) > anchor - 30 * 24 * H
    );

    const groups = [];

    for (const r of visible) {
        if (!r.anomaly) {
            continue;
        }

        let g = groups.at(-1);

        if (!g || Date.parse(r.time) - Date.parse(g.end) > H) {
            g = {
                id: `${id}-${groups.length + 1}`,
                time: r.time,
                end: r.time,
                points: []
            };

            groups.push(g);
        }

        g.points.push(r);
        g.end = r.time;
    }

    const anomalies = groups.reverse().map(g => ({
        id: g.id,
        time: g.time,
        end: g.end,
        title: 'Przekroczenie Q97',
        deviation: round(
            Math.max(
                ...g.points.map(
                    r => (r.przepompowano_m3 / r.prognoza - 1) * 100
                )
            )
        ),
        ponad_q97: round(
            g.points.reduce(
                (s, r) => s + r.przepompowano_m3 - r.q97,
                0
            )
        ),
        description: 'Przekroczenie Q97 wskazuje odchylenie od przewidywanego pompowania. Nie określa przyczyny, stanu urządzeń, kosztu naprawy ani możliwych oszczędności.'
    }));

    return {
        schemaVersion: 2,
        source: 'demo',
        asOf: new Date(anchor).toISOString(),
        station,
        stations,
        pricePerKwh: 1,
        history: business.detail,
        forecast: [],
        anomalies
    };
}