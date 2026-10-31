import { describe, expect, it } from 'vitest';
import { CURATED_LABELS, type Curated, type Example, type Manifest, type Table } from '../data/schema';

const EXPORTED_RANKS = 30;

const manifests = import.meta.glob<Manifest>('../../public/data/{,mock/}manifest.json', { eager: true, import: 'default' });
const tables = import.meta.glob<Table>('../../public/data/{,mock/}tables/sub-*.json', { eager: true, import: 'default' });
const examples = import.meta.glob<Example>('../../public/data/{,mock/}examples/sub-*/*.json', {
  eager: true,
  import: 'default',
});

const bundleOf = (manifestPath: string): string => manifestPath.slice(0, -'manifest.json'.length);
const pad = (n: number, width: number): string => String(n).padStart(width, '0');
const tableOf = (bundle: string, subject: number): Table => tables[`${bundle}tables/sub-${pad(subject, 2)}.json`];
const exampleOf = (bundle: string, subject: number, i: number): Example =>
  examples[`${bundle}examples/sub-${pad(subject, 2)}/${pad(i, 3)}.json`];
const examplesIn = (bundle: string): [string, Example][] =>
  Object.entries(examples).filter(([path]) => path.startsWith(`${bundle}examples/`));
const subjectOf = (path: string): string => String(Number(path.match(/sub-(\d+)/)![1]));
const imageOf = (path: string): number => Number(path.match(/(\d+)\.json$/)![1]);

const finalSize = (example: Example): number => example.set_size[example.stopped_at - 1];
const topPick = (example: Example): number => example.top[example.stopped_at - 1][0][0];

const labelHolds = (manifest: Manifest, example: Example, { i, label }: Curated): boolean =>
  ({
    'clean stop': example.claim.kind === 'single' && example.truth_in_set,
    unsure: example.stopped_at === manifest.K && finalSize(example) > manifest.m,
    'hedged, top pick wrong': example.truth_in_set && finalSize(example) > 1 && topPick(example) !== i,
    missed: !example.truth_in_set,
  })[label];

const bundles = Object.entries(manifests).map(([path, manifest]) => [bundleOf(path), manifest] as const);

it('finds at least one exported bundle', () => {
  expect(bundles.length).toBeGreaterThan(0);
});

describe.each(bundles)('bundle %s', (bundle, manifest) => {
  it('has results in the shape the views read', () => {
    const { results } = manifest;
    expect(Object.keys(results.per_subject).sort()).toEqual(manifest.subjects.map(String).sort());
    for (const summary of Object.values(results.per_subject)) {
      for (const key of ['eval_risk', 'conservativeness', 'mean_stop', 'mean_set_size', 'h1_ratio'] as const) {
        expect(summary[key], key).toEqual(expect.any(Number));
      }
      expect(summary.h2a.cp_upper).toEqual(expect.any(Number));
      expect(Object.keys(summary.outcome_mix).sort()).toEqual(['category', 'full', 'list', 'single']);
      expect(summary.fixed_points.length).toBeGreaterThan(0);
      for (const [size, k] of summary.fixed_points) {
        expect(size).toEqual(expect.any(Number));
        expect(summary.fixed_eval_risk[String(k)], `fixed k = ${k}`).toEqual(expect.any(Number));
      }
    }
    expect(results.pvalues).toEqual(expect.objectContaining({ H1: expect.any(Number), H2b: expect.any(Number) }));
    expect(results.holm).toEqual(expect.objectContaining({ H1: expect.any(Boolean), H2b: expect.any(Boolean) }));
    expect(results.h2b_count).toEqual(expect.any(Number));
  });

  it('has a loss table per subject sized to the gallery and the dial', () => {
    for (const subject of manifest.subjects) {
      const table = tableOf(bundle, subject);
      expect(table.loss_count).toHaveLength(manifest.concepts.length);
      expect(table.loss_count[0]).toHaveLength(manifest.lam.length);
      expect(table.naive.a_true_u16[0]).toHaveLength(manifest.R);
      expect(table.naive.top3_u16[0][0]).toHaveLength(manifest.K);
    }
  });

  it('exports replays whose stop is the first repeat with a small enough answer', () => {
    const replays = examplesIn(bundle);
    expect(replays.length).toBeGreaterThan(0);
    for (const [path, example] of replays) {
      expect(example.set_size, path).toHaveLength(manifest.K);
      expect(example.top, path).toHaveLength(manifest.K);
      expect(example.eeg, path).toHaveLength(manifest.K);
      const small = example.set_size.findIndex((size) => size <= manifest.m);
      expect(example.stopped_at, path).toBe(small === -1 ? manifest.K : small + 1);
      expect(example.truth_in_set, path).toEqual(expect.any(Boolean));
      example.top.forEach((ranked, t) => {
        expect(ranked.length, path).toBeGreaterThanOrEqual(Math.min(EXPORTED_RANKS, example.set_size[t]));
        ranked.slice(1).forEach(([, p], k) => expect(p, path).toBeLessThanOrEqual(ranked[k][1]));
      });
    }
  });

  it('exports replays only from the evaluation half of the pinned split', () => {
    for (const [path] of examplesIn(bundle)) {
      expect(manifest.pinned[subjectOf(path)].eval, path).toContain(imageOf(path));
    }
  });

  it('curates a miss and an unsure case, each labelled by what its replay shows', () => {
    const labels = manifest.curated.map((curated) => curated.label);
    expect(labels).toContain('missed');
    expect(labels).toContain('unsure');
    for (const curated of manifest.curated) {
      expect(CURATED_LABELS).toContain(curated.label);
      const example = exampleOf(bundle, curated.subject, curated.i);
      expect(example, `${curated.subject}/${curated.i}`).toBeDefined();
      expect(labelHolds(manifest, example, curated), `${curated.label}: ${curated.subject}/${curated.i}`).toBe(true);
    }
  });
});
