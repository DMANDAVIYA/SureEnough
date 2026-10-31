const FLOAT_NOISE = 1e-9;

export const mean = (values: number[]): number => values.reduce((sum, value) => sum + value, 0) / values.length;

export const noiseTolerantCeil = (x: number): number => Math.ceil(x - FLOAT_NOISE);
