import type { Example, Manifest, Table } from '../data/schema';

const MOCK_DIR = 'data/mock';

const dataDir = (): string =>
  import.meta.env.VITE_DATA_DIR || (new URLSearchParams(globalThis.location?.search).has('mock') ? MOCK_DIR : 'data');

export const dataUrl = (path: string): string => `${import.meta.env.BASE_URL}${dataDir()}/${path}`;

const pad = (n: number, width: number): string => String(n).padStart(width, '0');

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(dataUrl(path));
  if (!res.ok) throw new Error(`HTTP ${res.status} loading ${path}`);
  return (await res.json()) as T;
}

export const loadManifest = (): Promise<Manifest> => getJson('manifest.json');

export const loadTable = (subject: number): Promise<Table> =>
  getJson(`tables/sub-${pad(subject, 2)}.json`);

export const loadExample = (subject: number, i: number): Promise<Example> =>
  getJson(`examples/sub-${pad(subject, 2)}/${pad(i, 3)}.json`);
