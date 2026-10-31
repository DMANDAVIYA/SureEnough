export function stackedOpacities(weights: number[]): number[] {
  let below = 0;
  return weights.map((weight) => {
    below += weight;
    return weight / below;
  });
}
