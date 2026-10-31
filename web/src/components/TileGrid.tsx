import type { Concept } from '../data/schema';
import { dataUrl } from '../lib/data';

const SHORTLIST = 12;

interface Props {
  concepts: Concept[];
  setSize: number | null;
  members: Set<number> | null;
}

export default function TileGrid({ concepts, setSize, members }: Props) {
  const count = setSize ?? concepts.length;
  const shortlist = members && count <= SHORTLIST ? concepts.filter((c) => members.has(c.i)) : [];
  return (
    <section className="panel tiles" aria-label="What it could be">
      <h2>What it could be</h2>
      <p className="panel-note">
        <strong className="tiles-count">{count}</strong> of {concepts.length} objects still possible
      </p>
      {shortlist.length > 0 && (
        <ul className="shortlist">
          {shortlist.map((c) => (
            <li key={c.i}>
              <img src={dataUrl(c.tile)} alt="" width={56} height={56} />
              <span>{c.name}</span>
            </li>
          ))}
        </ul>
      )}
      <ul className="tile-grid" aria-label="All objects">
        {concepts.map((c) => (
          <li key={c.i} className={members && !members.has(c.i) ? 'tile tile-out' : 'tile'}>
            <img src={dataUrl(c.tile)} alt={c.name} title={c.name} loading="lazy" width={64} height={64} />
          </li>
        ))}
      </ul>
    </section>
  );
}
