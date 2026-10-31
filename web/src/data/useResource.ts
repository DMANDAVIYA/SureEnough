import { useEffect, useState } from 'react';

type Loaded<T> = T | Error | null;

export const ready = <T>(loaded: Loaded<T>): T | null => (loaded instanceof Error ? null : loaded);

export function useResource<T>(key: string | null, load: () => Promise<T>): Loaded<T> {
  const [state, setState] = useState<{ key: string; value: T | Error } | null>(null);
  useEffect(() => {
    if (key === null) return;
    let live = true;
    load().then(
      (value) => live && setState({ key, value }),
      (error: unknown) => live && setState({ key, value: error instanceof Error ? error : new Error(String(error)) }),
    );
    return () => {
      live = false;
    };
  }, [key]);
  return state?.key === key ? state.value : null;
}
