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

  it('switches to the mock bundle when the page URL has ?mock', () => {
    vi.stubEnv('BASE_URL', '/SureEnough/');
    vi.stubGlobal('location', { search: '?mock' });
    expect(dataUrl('manifest.json')).toBe('/SureEnough/data/mock/manifest.json');
  });

  it('reads from VITE_DATA_DIR when it is set', () => {
    vi.stubEnv('BASE_URL', '/SureEnough/');
    vi.stubEnv('VITE_DATA_DIR', 'data/mock');
    expect(dataUrl('tables/sub-01.json')).toBe('/SureEnough/data/mock/tables/sub-01.json');
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
