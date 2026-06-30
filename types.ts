/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface ModelMetrics {
  Model: string;
  MAE: number;
  MSE: number;
  RMSE: number;
  R2Score: number;
  AdjustedR2: number;
}

export interface FeatureImportance {
  Feature: string;
  Importance: number;
}

export interface DatasetRow {
  MedInc: number;
  HouseAge: number;
  AveRooms: number;
  AveBedrms: number;
  Population: number;
  AveOccup: number;
  Latitude: number;
  Longitude: number;
  Price: number;
}

export interface TrainingLog {
  timestamp: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
}

export interface PredictionInput {
  MedInc: number;
  HouseAge: number;
  AveRooms: number;
  AveBedrms: number;
  Population: number;
  AveOccup: number;
  Latitude: number;
  Longitude: number;
}

export interface TuningResult {
  paramName: string;
  paramGrid: string;
  bestValue: string;
  explanation: string;
}
