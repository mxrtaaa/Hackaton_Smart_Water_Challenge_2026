import test from 'node:test';
import assert from 'node:assert/strict';
import net from 'node:net';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';

const serverPath = new URL('../server.mjs', import.meta.url);

function start(env) {
  const child = spawn(process.execPath, [fileURLToPath(serverPath)], {
    env, stdio: ['ignore', 'pipe', 'pipe'],
  });
  let output = '';
  child.stdout.on('data', chunk => { output += chunk; });
  child.stderr.on('data', chunk => { output += chunk; });
  return { child, output: () => output };
}

async function waitForAddress(process) {
  const deadline = Date.now() + 5000;
  while (Date.now() < deadline) {
    const match = process.output().match(/Smart Water UI: (http:\/\/127\.0\.0\.1:\d+)/);
    if (match) return match[1];
    if (process.child.exitCode !== null) throw new Error(process.output());
    await new Promise(resolve => setTimeout(resolve, 20));
  }
  throw new Error(`Serwer nie wystartował: ${process.output()}`);
}

test('Zajęty domyślny port: start na kolejnym porcie i działające API', async () => {
  const blocker = net.createServer();
  try {
    await new Promise((resolve, reject) => { blocker.once('error', reject); blocker.listen(4173, '127.0.0.1', resolve); });
  } catch (error) {
    if (error.code !== 'EADDRINUSE') throw error; // Inny proces już blokuje port.
  }
  const env = { ...globalThis.process.env };
  delete env.PORT;
  delete env.UI_DATA_FILE;
  const process = start(env);
  try {
    const address = await waitForAddress(process);
    const port = Number(new URL(address).port);
    assert.ok(port > 4173 && port <= 4183);
    const response = await fetch(`${address}/api/dashboard?station=chorzow`);
    assert.equal(response.status, 200);
    const dashboard = await response.json();
    assert.equal(dashboard.schemaVersion, 2);
    assert.equal(dashboard.source, 'processed-models');
    assert.equal(dashboard.businessEngine, 'python');
    assert.ok(dashboard.stations.every(s=>s.id==='chorzow'));
    assert.ok(dashboard.forecast.every(r=>r.data_kind==='prediction'&&r.przepompowano_m3===null&&r.energia_kwh===null));
    assert.ok(dashboard.businessByRange['30'].summary.godziny_wycenione > 0);
    assert.ok(dashboard.history.some(r=>r.energia_q97!==null));
    const localDay=t=>new Intl.DateTimeFormat('en-CA',{timeZone:'Europe/Warsaw',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(t));
    const selectedDay=localDay(dashboard.asOf), selectedRows=dashboard.history.filter(r=>localDay(r.time)===selectedDay);
    for(const energyBasis of ['measured','predicted']){
      const response=await fetch(`${address}/api/cost-scenarios`,{
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({station:'chorzow',days:14,period:'history',metric:'energy',startDate:selectedDay,endDate:selectedDay,energyBasis,prices:{lowPrice:1,basePrice:2,highPrice:3}}),
      });
      assert.equal(response.status,200);
      const values=await response.json(), field=energyBasis==='predicted'?'energia_prognoza':'energia_kwh';
      const volume=energyBasis==='predicted'?'prognoza':'przepompowano_m3';
      assert.ok(Math.abs(values[1].valuation.totalCostZl-selectedRows.reduce((s,r)=>s+(r[field]??0),0)*2)<1e-8);
      assert.ok(Math.abs(values[1].valuation.pumpingM3-selectedRows.reduce((s,r)=>s+(r[volume]??0),0))<1e-8);
      assert.equal(values[1].valuation.dataKind,energyBasis==='predicted'?'prediction':'measurement');
      assert.equal(values[1].valuation.expectedHours,selectedRows.length);
    }
    const prices = await fetch(`${address}/api/cost-scenarios`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({station:'chorzow',days:30,prices:{lowPrice:.5,basePrice:1,highPrice:2}}),
    });
    assert.equal(prices.status, 200);
    const scenarios = await prices.json();
    assert.equal(scenarios[1].summary.koszt_pelnych_epizodow_zl, dashboard.businessByRange['30'].summary.koszt_pelnych_epizodow_zl);
    assert.equal(scenarios[2].summary.koszt_pelnych_epizodow_zl, 2 * scenarios[1].summary.koszt_pelnych_epizodow_zl);
    if (dashboard.forecast.length) {
      const future = await fetch(`${address}/api/cost-scenarios`, {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({station:'chorzow',days:30,period:'forecast',metric:'energy',prices:{lowPrice:1,basePrice:2,highPrice:3}}),
      });
      assert.equal(future.status, 200);
      const valuations = await future.json();
      const predictedEnergy = dashboard.forecast.reduce((sum,r)=>sum+(r.energia_prognoza??0),0);
      assert.ok(Math.abs(valuations[1].valuation.totalCostZl-2*predictedEnergy)<1e-7);
      assert.equal(valuations[1].valuation.dataKind,'prediction');
    }
    const invalid = await fetch(`${address}/api/cost-scenarios`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({station:'chorzow',days:30,prices:{lowPrice:-1,basePrice:1,highPrice:2}}),
    });
    assert.equal(invalid.status, 400);
  } finally {
    const closed = once(process.child, 'close');
    process.child.kill();
    await closed;
    if (blocker.listening) await new Promise(resolve => blocker.close(resolve));
  }
});

test('Jawnie wskazany zajęty port: czytelny komunikat zamiast nieobsłużonego wyjątku', async () => {
  const blocker = net.createServer();
  await new Promise(resolve => blocker.listen(0, '127.0.0.1', resolve));
  const process = start({ ...globalThis.process.env, PORT: String(blocker.address().port) });
  try {
    const [code] = await once(process.child, 'close');
    assert.equal(code, 1);
    assert.match(process.output(), /port .* jest zajęty/);
    assert.doesNotMatch(process.output(), /Unhandled|throw er/);
  } finally {
    if (process.child.exitCode === null) process.child.kill();
    await new Promise(resolve => blocker.close(resolve));
  }
});

test('Weteranów tab uses approved assets and date-based costs even with demo/snapshot configured', async () => {
  const process = start({...globalThis.process.env,PORT:'0',UI_SOURCE:'demo',UI_DATA_FILE:'missing-snapshot-that-must-not-be-used.json'});
  try {
    const address = await waitForAddress(process);
    const page = await fetch(`${address}/?station=ruda-weteranow`);
    const html = await page.text();
    assert.ok(html.includes('data-station="ruda-weteranow"'));
    assert.ok(html.includes('id="calc-start"') && html.includes('id="calc-end"'));
    const response = await fetch(`${address}/api/dashboard?station=ruda-weteranow`);
    assert.equal(response.status,200);
    const data = await response.json();
    assert.equal(data.source,'weteranow-models');
    assert.equal(data.station.id,'ruda-weteranow');
    assert.ok(data.history.some(r=>r.energia_kwh!==null&&r.energia_prognoza!==null&&r.energia_q97!==null));
    assert.ok(data.provenance.files.every(x=>!x.file.includes('reports')&&!x.file.includes('zmp')));
    const localDay=t=>new Intl.DateTimeFormat('en-CA',{timeZone:'Europe/Warsaw',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(t));
    const date=localDay(data.asOf),rows=data.history.filter(r=>localDay(r.time)===date);
    const cost = await fetch(`${address}/api/cost-scenarios`,{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({station:'ruda-weteranow',days:14,period:'history',metric:'energy',startDate:date,endDate:date,energyBasis:'predicted',prices:{lowPrice:1,basePrice:2,highPrice:3}}),
    });
    assert.equal(cost.status,200);
    const result=await cost.json();
    assert.equal(result[1].valuation.dataKind,'prediction');
    const expected=rows.reduce((sum,r)=>sum+(r.energia_prognoza??0),0);
    assert.ok(Math.abs(result[1].valuation.totalCostZl-expected*2)<1e-8);
    if(data.forecast.length){
      assert.equal(data.forecast.length,720);
      assert.ok(data.forecast.every(r=>r.przepompowano_m3===null&&r.energia_kwh===null&&r.data_kind==='prediction'));
      const date=localDay(data.forecast[0].time);
      const futureCost=await fetch(`${address}/api/cost-scenarios`,{
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({station:'ruda-weteranow',days:30,period:'forecast',metric:'energy',startDate:date,endDate:date,prices:{lowPrice:1,basePrice:2,highPrice:3}}),
      });
      assert.equal(futureCost.status,200);
      const future=await futureCost.json();
      const energy=data.forecast.filter(r=>localDay(r.time)===date).reduce((sum,r)=>sum+(r.energia_prognoza??0),0);
      assert.ok(Math.abs(future[1].valuation.totalCostZl-energy*2)<1e-8);
    }
  }finally{
    const closed=once(process.child,'close');process.child.kill();await closed;
  }
});
