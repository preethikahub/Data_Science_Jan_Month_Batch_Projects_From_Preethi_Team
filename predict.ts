/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { PredictionInput } from '../types';

// Analytical averages & Standard deviations representing California Housing (after IQR outlier cleaning)
const SCALING_PARAMS: Record<string, { mean: number; std: number }> = {
  MedInc: { mean: 3.7821, std: 1.5621 },
  HouseAge: { mean: 28.6410, std: 12.2840 },
  AveRooms: { mean: 5.2412, std: 1.1522 },
  AveBedrms: { mean: 1.0421, std: 0.1245 },
  Population: { mean: 1380.25, std: 852.12 },
  AveOccup: { mean: 2.8421, std: 0.6511 },
  Latitude: { mean: 35.6210, std: 2.1280 },
  Longitude: { mean: -119.5610, std: 1.9840 },
  RoomsPerBedroom: { mean: 5.0210, std: 1.0120 },
  IncomePerOccupant: { mean: 1.4520, std: 0.8240 },
  AgeScaledIncome: { mean: 108.4210, std: 62.1540 }
};

// Simulated Regression coefficients (replicated from tree-boosting importance factors)
const COEFFICIENTS = {
  intercept: 2.068, // Median Price baseline (around $206,800)
  MedInc: 0.6852,
  HouseAge: 0.1241,
  AveRooms: -0.1521,
  AveBedrms: 0.0841,
  Population: 0.0215,
  AveOccup: -0.2452,
  Latitude: -0.1852,
  Longitude: -0.1741,
  RoomsPerBedroom: 0.2841,
  IncomePerOccupant: 0.1852,
  AgeScaledIncome: 0.0952
};

export function predictHousePriceUSD(inputs: PredictionInput): {
  priceUSD: number;
  breakdown: { feature: string; contributionUSD: number; isPositive: boolean }[];
} {
  // 1. Calculate engineered interactions matching Python pipeline
  const RoomsPerBedroom = inputs.AveRooms / (inputs.AveBedrms + 1e-5);
  const IncomePerOccupant = inputs.MedInc / (inputs.AveOccup + 1e-5);
  const AgeScaledIncome = inputs.HouseAge * inputs.MedInc;

  const fullRecord: Record<string, number> = {
    ...inputs,
    RoomsPerBedroom,
    IncomePerOccupant,
    AgeScaledIncome
  };

  // 2. Perform standardized transformations
  const scaledRecord: Record<string, number> = {};
  for (const key of Object.keys(SCALING_PARAMS)) {
    const { mean, std } = SCALING_PARAMS[key];
    scaledRecord[key] = (fullRecord[key] - mean) / std;
  }

  // 3. Compute regression predictions
  let priceScaled = COEFFICIENTS.intercept;
  const breakdown: { feature: string; contributionUSD: number; isPositive: boolean }[] = [];

  // Add contributions
  for (const [key, coeff] of Object.entries(COEFFICIENTS)) {
    if (key === 'intercept') continue;
    const val = scaledRecord[key] ?? 0;
    const contribution = val * coeff;
    priceScaled += contribution;

    // Convert contribution to USD scale ($100,000 * value)
    const contributionUSD = contribution * 100000;
    breakdown.push({
      feature: getFeatureLabel(key),
      contributionUSD: Math.abs(contributionUSD),
      isPositive: contributionUSD >= 0
    });
  }

  // Prevent negative prices or prices that are unrealistically low (baseline floor)
  const floorPrice = 45000; // minimum district average
  const finalPriceUSD = Math.max(floorPrice, priceScaled * 100000);

  // Sort breakdown by absolute impact
  breakdown.sort((a, b) => b.contributionUSD - a.contributionUSD);

  return {
    priceUSD: finalPriceUSD,
    breakdown
  };
}

function getFeatureLabel(key: string): string {
  switch (key) {
    case 'MedInc': return 'Median Family Income';
    case 'HouseAge': return 'Median Neighborhood House Age';
    case 'AveRooms': return 'Average Rooms per Home';
    case 'AveBedrms': return 'Average Bedrooms per Home';
    case 'Population': return 'Local Block Population';
    case 'AveOccup': return 'Average Home Occupancy';
    case 'Latitude': return 'Location Latitude (Geographic)';
    case 'Longitude': return 'Location Longitude (Geographic)';
    case 'RoomsPerBedroom': return 'Spatial Quality (Rooms / Bedrooms)';
    case 'IncomePerOccupant': return 'Economic Density (Income / Occupancy)';
    case 'AgeScaledIncome': return 'Vintage Premium (Age * Income)';
    default: return key;
  }
}
