export default function Pips({ total, open }: { total: number; open: number }) {
  return (
    <span className="pips" aria-hidden="true">
      {Array.from({ length: total }, (_, k) => (
        <span key={k} className={k < total - open ? 'pip' : 'pip pip-open'} />
      ))}
    </span>
  );
}
