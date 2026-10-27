import { callPython } from './python-bridge.mjs';

// Read-only inference from exported models and hourly project reports.
// Model or dependency errors are reported; they never fall back to demo data.
export const getModelDashboard = station => callPython({action:'dashboard',station});
export const getModelCosts = (station,days,prices,period='history',metric='wet_excess',options={}) => callPython({action:'costs',station,days,prices,period,metric,...options});
