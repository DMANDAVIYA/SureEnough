export function mapInBackground<T, R>(items: T[], work: (item: T) => R, done: (results: R[]) => void): () => void {
  const results: R[] = [];
  let timer: ReturnType<typeof setTimeout>;
  const next = () => {
    if (results.length === items.length) return done(results);
    results.push(work(items[results.length]));
    timer = setTimeout(next, 0);
  };
  timer = setTimeout(next, 0);
  return () => clearTimeout(timer);
}
