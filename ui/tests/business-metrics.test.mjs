import test from 'node:test';
import assert from 'node:assert/strict';
import {classifyWeatherPeriods,calculateDryBaseline,calculateExcess,summarizeDetail} from '../public/business-metrics.mjs';
import {getDashboard,stations} from '../demo-source.mjs';
const time=i=>new Date(Date.parse('2026-01-01T00:00:00Z')+i*3600000).toISOString();
test('Klasyfikacja wymaga kompletnych 48 h; zero opadów jest poprawną wartością',()=>{
  const rows=Array.from({length:50},(_,i)=>({time:time(i),opad_mm:i===49?1:0})),r=classifyWeatherPeriods(rows);
  assert.equal(r[46].okres,'nieokreslony');assert.equal(r[47].okres,'suchy');assert.equal(r[49].okres,'mokry');
});
test('Brakujące godziny są odtwarzane jako braki, nie zera',()=>{
  const r=classifyWeatherPeriods([{time:time(0),opad_mm:0},{time:time(2),opad_mm:0}]);assert.equal(r.length,3);assert.equal(r[1].opad_mm,null);assert.equal(r[1].przepompowano_m3,null);
});
test('Duplikaty, czas bez strefy i ujemne opady są odrzucane',()=>{
  assert.throws(()=>classifyWeatherPeriods([{time:time(0)},{time:time(0)}]));assert.throws(()=>classifyWeatherPeriods([{time:'2026-01-01T12:00:00'}]));assert.throws(()=>classifyWeatherPeriods([{time:time(0),opad_mm:-1}]));
});
test('Baseline wymaga 5 próbek, zachowuje medianę i kolejność fallback',()=>{
  const rows=Array.from({length:5},(_,i)=>({time:time(i),okres:'suchy',miesiac:1,godzina:1,weekend:0,przepompowano_m3:i*10,energia_kwh:i*2}));
  const r=calculateDryBaseline([...rows,{okres:'mokry',miesiac:2,godzina:1,weekend:0}]);assert.equal(r[0].bazowe_m3,20);assert.equal(r[0].bazowe_kwh,4);assert.equal(r[0].baseline_zrodlo,'miesiac+godzina+weekend');assert.equal(r[5].baseline_zrodlo,'godzina+weekend');assert.throws(()=>calculateDryBaseline(rows.slice(0,4)));
});
test('Nadwyżka zachowuje znak, wyceniane są wyłącznie kompletne mokre godziny',()=>{
  const r=calculateExcess([{okres:'mokry',przepompowano_m3:5,energia_kwh:2,bazowe_m3:10,bazowe_kwh:4},{okres:'suchy',przepompowano_m3:5,energia_kwh:2,bazowe_m3:10,bazowe_kwh:4},{okres:'mokry',przepompowano_m3:null,energia_kwh:2,bazowe_m3:10,bazowe_kwh:4}],2);
  assert.equal(r[0].nadwyzka_m3,-5);assert.equal(r[0].koszt_nadwyzki_zl,-4);assert.equal(r[1].nadwyzka_kwh,null);assert.equal(r[2].nadwyzka_kwh,null);
});
test('Epizody: obcięcie netto po sumowaniu, bez wyceny niekompletnych; bilans miesięczny zachowuje znak',()=>{
  const r=[{time:time(0),okres:'mokry',nadwyzka_kwh:10,nadwyzka_m3:20},{time:time(1),okres:'mokry',nadwyzka_kwh:-7,nadwyzka_m3:-10},{time:time(2),okres:'suchy'},{time:time(3),okres:'mokry',nadwyzka_kwh:null,nadwyzka_m3:null},{time:time(4),okres:'suchy'},{time:time(5),okres:'mokry',nadwyzka_kwh:-5,nadwyzka_m3:-8}];
  const s=summarizeDetail(r,2);assert.equal(s.episodes[0].koszt_zl,6);assert.equal(s.episodes[1].koszt_zl,0);assert.equal(s.episodes[2].koszt_zl,null);assert.equal(s.summary.pelne_epizody,2);assert.equal(s.summary.koszt_pelnych_epizodow_zl,6);assert.equal(s.monthly[0].bilans_zl,-4);assert.equal(s.monthly[0].pokrycie,.75);
});
test('Siatka godzinowa podczas zmiany czasu jest absolutna, godziny lokalne są polskie',()=>{
  const r=classifyWeatherPeriods([{time:'2026-10-25T00:00:00Z',opad_mm:0},{time:'2026-10-25T01:00:00Z',opad_mm:0}]);assert.equal(r.length,2);assert.equal(r[0].godzina,2);assert.equal(r[1].godzina,2);
});
test('Demo nie zwraca fikcyjnych prognoz, statusów pomp ani kosztów interwencji',()=>{
  for(const station of stations){
    const d=getDashboard(station.id);
    assert.equal(d.schemaVersion,2);assert.equal(d.source,'demo');assert.deepEqual(d.forecast,[]);
    assert.ok(d.anomalies.length>=3);assert.ok(d.history.some(r=>r.baseline_n>=5));assert.ok(d.history.some(r=>r.przepompowano_m3===null));
    assert.ok(d.history.every(r=>Date.parse(r.time)<=Date.parse(d.asOf)));
    assert.ok(d.history.every(r=>!r.anomaly||r.przepompowano_m3>r.q97));
    assert.ok(!('telemetry' in d));assert.ok(!('interventions' in d));assert.ok(!('pumps' in d.station));
    for(const a of d.anomalies){assert.ok(!('dailyCost' in a));assert.ok(!('extraCost' in a));assert.ok(!('status' in a));}
  }
});
