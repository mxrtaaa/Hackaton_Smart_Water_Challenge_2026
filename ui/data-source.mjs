import { getDashboard as getDemoDashboard } from './demo-source.mjs';
import { getModelDashboard } from './real-data-source.mjs';

export function getDashboard(station = 'chorzow') {
    if (station === 'ruda-weteranow') {
        return getModelDashboard(station);
    }

    return process.env.UI_SOURCE === 'demo'
        ? getDemoDashboard(station)
        : getModelDashboard(station);
}