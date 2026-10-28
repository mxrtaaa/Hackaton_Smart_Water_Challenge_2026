export function historyDays(range) { 
  return ['7','14','30'].includes(String(range))?Number(range):30; 
}

export function selectHistory(data,days) {

  const end=Date.parse(data.asOf),start=end-days*86400000;
  return data.history.filter(r=>Date.parse(r.time)>start&&Date.parse(r.time)<=end);
}

export function lastMeasuredHour(data) {
  const time=Date.parse(data.asOf);
  const row=data.history.find(r=>Date.parse(r.time)===time&&r.data_kind!=='prediction');
  const measured=value=>typeof value==='number'&&Number.isFinite(value)?value:null;
  
  return {time:data.asOf,pumpingM3:measured(row?.przepompowano_m3),energyKwh:measured(row?.energia_kwh)};
}

export function selectForecast(data) {
  const rows=data.forecast||[],asOf=Date.parse(data.asOf);

  for(const r of rows){

    if(r.data_kind!=='prediction'|| r.przepompowano_m3!==null
      || r.energia_kwh!==null|| !Number.isFinite(Date.parse(r.time))
      || Date.parse(r.time)<=asOf){

      throw new Error('Nieprawidłowa prognoza: przyszłe wiersze mogą zawierać wyłącznie predykcje, bez pomiarów.');
    }
  }

  return rows;
}
