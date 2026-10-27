import test from 'node:test';
import assert from 'node:assert/strict';
import { calculateCostScenarios } from '../public/economics.mjs';
import { calculateExcess, summarizeDetail } from '../public/business-metrics.mjs';
const row=(i,energy)=>({time:new Date(Date.parse('2026-01-01T00:00:00Z')+i*3600000).toISOString(),okres:'mokry',przepompowano_m3:100,energia_kwh:energy,bazowe_m3:80,bazowe_kwh:10});
const prices={lowPrice:.5,basePrice:1,highPrice:2};
test('Wycena używa dokładnie calculateExcess i summarizeDetail, bez ROI',()=>{
  const rows=[row(0,20),row(1,7)];
  const scenarios=calculateCostScenarios(rows,prices);
  assert.deepEqual(scenarios[1].summary,summarizeDetail(calculateExcess(rows,1),1).summary);
  assert.deepEqual(scenarios.map(s=>s.summary.koszt_pelnych_epizodow_zl),[3.5,7,14]);
  for(const s of scenarios){assert.ok(!('repairNet' in s));assert.ok(!('totalPayback' in s));assert.ok(!('expectedGross' in s));}
});
test('Nie obcinamy nadwyżek godzinowych przed zsumowaniem epizodu',()=>{
  const s=calculateCostScenarios([row(0,20),row(1,0)],prices);
  assert.equal(s[1].summary.koszt_pelnych_epizodow_zl,0);assert.equal(s[1].monthly[0].bilans_zl,0);
});
test('Niekompletny epizod pozostaje niewyceniony we wszystkich wariantach',()=>{
  const scenarios=calculateCostScenarios([row(0,20),row(1,null)],prices);
  for(const s of scenarios){assert.equal(s.episodes[0].koszt_zl,null);assert.equal(s.summary.pelne_epizody,0);assert.equal(s.summary.godziny_wycenione,1);assert.equal(s.monthly[0].pokrycie,.5);}
});
test('Ujemny bilans miesięczny zachowuje znak',()=>{
  const s=calculateCostScenarios([row(0,5)],prices);assert.equal(s[1].monthly[0].bilans_zl,-5);assert.equal(s[1].summary.koszt_pelnych_epizodow_zl,0);
});
test('Domyślny zakres może być równy cenie źródłowej, bez arbitralnych widełek',()=>{
  const s=calculateCostScenarios([row(0,20)],{lowPrice:1,basePrice:1,highPrice:1});assert.deepEqual(s.map(x=>x.summary.koszt_pelnych_epizodow_zl),[10,10,10]);
});
test('Walidacja braków, ujemnych cen i kolejności scenariuszy; zero jest poprawne',()=>{
  for(const patch of [{lowPrice:NaN},{highPrice:Infinity},{basePrice:-1},{basePrice:3},{lowPrice:1.5}])assert.throws(()=>calculateCostScenarios([],{...prices,...patch}));
  assert.equal(calculateCostScenarios([row(0,20)],{lowPrice:0,basePrice:0,highPrice:0})[1].summary.koszt_pelnych_epizodow_zl,0);
});
