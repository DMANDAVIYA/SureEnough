import { afterEach, describe, expect, it, vi } from 'vitest';
import config from '../../vite.config';
import { dataUrl, loadExample, loadManifest, loadTable } from './data';

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe('dataUrl', () => {
  it('resolves under the Pages sub-path', () => {
    vi.stubEnv('BASE_URL', '/SureEnough/');
    expect(dataUrl('manifest.json').endsWith('/SureEnough/data/manifest.json')).toBe(true);
  });

  it('is built with the /SureEnough/ base', () => {
    expect(config.base).toBe('/SureEnough/');
  });
});

describe('loaders', () => {
  it('fetch zero-padded export paths', async () => {
    vi.stubEnv('BASE_URL', '/SureEnough/');
    const fetchMock = vi.fn(async (_url: string) => new Response('{}'));
    vi.stubGlobal('fetch', fetchMock);
    await loadManifest();
    await loadTable(3);
    await loadExample(3, 7);
    expect(fetchMock.mock.calls.map((c) => c[0])).toEqual([
      '/SureEnough/data/manifest.json',
      '/SureEnough/data/tables/sub-03.json',
      '/SureEnough/data/examples/sub-03/007.json',
    ]);
  });

  it('reject on HTTP errors', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 404 })));
    await expect(loadManifest()).rejects.toThrow('404');
  });
});
