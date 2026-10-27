import { calculateExcess, summarizeDetail } from './business-metrics.mjs';

// Wycena historyczna. Cena jest jedynym założeniem scenariusza.

export function validatePrices({ lowPrice, basePrice, highPrice }) {
  for (const price of [lowPrice, basePrice, highPrice]) {
    if (typeof price !== 'number' || !Number.isFinite(price) || price < 0) {
      throw new Error('Uzupełnij każdą cenę nieujemną liczbą.');
    }
  }
  if (lowPrice > basePrice || basePrice > highPrice) {
    throw new Error('Zachowaj kolejność cen: dolna ≤ bazowa ≤ górna.');
  }
}

export function calculateCostScenarios(rows, { lowPrice, basePrice, highPrice }) {
  validatePrices({lowPrice,basePrice,highPrice});
  return [ ['Dolny', lowPrice], ['Bazowy', basePrice], ['Górny', highPrice] ].map(([name, price]) => {
    const detail = calculateExcess(rows, price);
    return { name, price, ...summarizeDetail(detail, price) };
  });
}
