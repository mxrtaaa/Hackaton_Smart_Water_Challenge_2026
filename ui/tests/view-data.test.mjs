import test from 'node:test';
import assert from 'node:assert/strict';
import {historyDays,selectHistory,selectForecast,lastMeasuredHour} from '../public/view-data.mjs';
test('History windows anchor to the last available measurement, including 14 days',()=>{
  const end=Date.parse('2026-10-12T22:00:00Z');
  const data={asOf:new Date(end).toISOString(),history:Array.from({length:720},(_,i)=>({time:new Date(end-(719-i)*3600000).toISOString()}))};
  assert.equal(selectHistory(data,7).length,168);assert.equal(selectHistory(data,14).length,336);assert.equal(historyDays('future'),30);
});
test('Forecast never contains actual measurements or historical dates',()=>{
  const asOf='2026-10-12T22:00:00Z';
  const row={time:'2026-10-28T00:00:00Z',data_kind:'prediction',przepompowano_m3:null,energia_kwh:null,prognoza:10};
  assert.equal(selectForecast({asOf,forecast:[row]}).length,1);
  for(const change of [{przepompowano_m3:10},{energia_kwh:5},{data_kind:'measurement'},{time:asOf}])assert.throws(()=>selectForecast({asOf,forecast:[{...row,...change}]}));
  assert.deepEqual(selectForecast({asOf,forecast:[]}),[]);
});
test('Last-hour cards keep real zeros and never substitute an older or predicted value',()=>{
  const asOf='2026-10-12T22:00:00Z';
  const data={asOf,history:[{time:'2026-10-12T21:00:00Z',przepompowano_m3:80,energia_kwh:20},{time:asOf,przepompowano_m3:0,energia_kwh:null}],forecast:[{time:'2026-10-28T00:00:00Z',prognoza:100,energia_prognoza:30}]};
  assert.deepEqual(lastMeasuredHour(data),{time:asOf,pumpingM3:0,energyKwh:null});
  data.history.pop();assert.deepEqual(lastMeasuredHour(data),{time:asOf,pumpingM3:null,energyKwh:null});
});
