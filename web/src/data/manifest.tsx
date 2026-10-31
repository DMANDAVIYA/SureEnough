import { createContext, type ReactNode, useContext } from 'react';
import { loadManifest } from '../lib/data';
import type { Manifest } from './schema';
import { useResource } from './useResource';

const ManifestContext = createContext<Manifest | null>(null);

export function ManifestProvider({ children }: { children: ReactNode }) {
  const manifest = useResource('manifest', loadManifest);
  if (manifest instanceof Error) {
    return (
      <p className="status" role="alert">
        The replay data did not load ({manifest.message}). Export it into web/public/data, or add ?mock to the address to
        use the synthetic bundle.
      </p>
    );
  }
  if (manifest === null) return <p className="status">Loading the replays…</p>;
  return <ManifestContext.Provider value={manifest}>{children}</ManifestContext.Provider>;
}

export function useManifest(): Manifest {
  const manifest = useContext(ManifestContext);
  if (manifest === null) throw new Error('useManifest needs a ManifestProvider');
  return manifest;
}
