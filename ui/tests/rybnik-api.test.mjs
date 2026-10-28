import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';

test('PZ1 and PNB1 have separate final models, histories, future costs and tabs', { timeout: 120000 }, async () => {
  const child = spawn(process.execPath, [fileURLToPath(new URL('../server.mjs', import.meta.url))], {
    env: { ...process.env, PORT: '0', UI_SOURCE: 'demo', UI_DATA_FILE: 'missing-snapshot.json' },
    stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true,
  });
  let output = '';
  child.stdout.on('data', chunk => { output += chunk; });
  child.stderr.on('data', chunk => { output += chunk; });
  try {
    const address = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(output || 'Server did not start')), 5000);
      const read = () => {
        const match = output.match(/Smart Water UI: (http:\/\/127\.0\.0\.1:\d+)/);
        if (match) { clearTimeout(timer); resolve(match[1]); }
      };
      child.stdout.on('data', read);
      child.once('error', error => { clearTimeout(timer); reject(error); });
      child.once('close', () => { clearTimeout(timer); reject(new Error(output)); });
      read();
    });
    const day = t => new Intl.DateTimeFormat('en-CA', { timeZone: 'Europe/Warsaw', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date(t));
    for (const [station, object] of [['rybnik_pz1', 'pz1'], ['rybnik_pnb1', 'pnb1']]) {
      const html = await (await fetch(`${address}/?station=${station}`)).text();
      assert.ok(html.includes(`data-station="${station}"`));
      assert.ok(html.includes(`value="${station}"`));
      const response = await fetch(`${address}/api/dashboard?station=${station}`);
      assert.equal(response.status, 200);
      const data = await response.json();
      assert.equal(data.station.id, station);
      assert.equal(data.source, 'rybnik-final-models');
      assert.equal(data.businessEngine, 'python');
      assert.equal(data.history.length, 720);
      assert.ok(data.history.some(r => r.energia_prognoza !== null && r.energia_q97 !== null));
      assert.ok(data.history.some(r => r.energia_kwh !== null));
      assert.notEqual(data.history.at(-1).energia_kwh, null);
      assert.notEqual(data.history.at(-1).przepompowano_m3, null);
      assert.ok(data.provenance.models.every(m => m.path.includes('final') && m.path.includes(`_${object}_`)));
      assert.ok(data.provenance.files.some(f => f.file.includes(`godzinowe-${object}-final.parquet`)));
      assert.ok(data.provenance.files.some(f => f.file.includes('suma_opadow.parquet')));
      assert.equal(day(data.asOf), object === 'pz1' ? '2025-10-01' : '2026-09-28');
      const date = day(data.asOf), rows = data.history.filter(r => day(r.time) === date);
      const cost = await fetch(`${address}/api/cost-scenarios`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ station, days: 14, period: 'history', metric: 'energy',
          startDate: date, endDate: date, energyBasis: 'predicted',
          prices: { lowPrice: 1, basePrice: 2, highPrice: 3 } }),
      });
      assert.equal(cost.status, 200);
      const result = await cost.json();
      assert.equal(result[1].valuation.dataKind, 'prediction');
      const expected = rows.reduce((sum, r) => sum + (r.energia_prognoza ?? 0), 0);
      assert.ok(Math.abs(result[1].valuation.totalCostZl - expected * 2) < 1e-8);
      if (data.weatherForecast.status !== 'unavailable') {
        assert.equal(data.forecast.length, data.weatherForecast.expectedHours);
        assert.equal(data.forecast.length, 720);
        assert.equal(Date.parse(data.forecast.at(-1).time) - Date.parse(data.forecast[0].time), 719 * 3600000);
        assert.equal(data.weatherForecast.ensembleAggregation, 'none');
        assert.equal(data.weatherForecast.ensembleMember, 0);
        assert.ok(data.forecast.every(r => r.data_kind === 'prediction' && r.przepompowano_m3 === null && r.energia_kwh === null));
        const date = day(data.forecast[0].time);
        const cost = await fetch(`${address}/api/cost-scenarios`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ station, days: 30, period: 'forecast', metric: 'energy',
            startDate: date, endDate: date, prices: { lowPrice: 1, basePrice: 2, highPrice: 3 } }),
        });
        assert.equal(cost.status, 200);
        const result = await cost.json();
        const expected = data.forecast.filter(r => day(r.time) === date).reduce((sum, r) => sum + (r.energia_prognoza ?? 0), 0);
        assert.ok(Math.abs(result[1].valuation.totalCostZl - expected * 2) < 1e-8);
      }
    }
  } finally {
    if (child.exitCode === null) {
      const closed = once(child, 'close');
      child.kill();
      await closed;
    }
  }
});
