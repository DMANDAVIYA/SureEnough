import type { ClaimKind, OutcomeMix } from '../data/schema';

const KINDS: [ClaimKind, string][] = [
  ['single', 'Single object'],
  ['category', 'Category'],
  ['list', 'List'],
  ['full', 'Full set'],
];

interface Props {
  title: string;
  mix: OutcomeMix;
  current: ClaimKind | null;
}

export default function OutcomeCounters({ title, mix, current }: Props) {
  return (
    <aside className="counters" aria-label="Outcome counters">
      <p className="counters-title">{title}</p>
      <ul>
        {KINDS.map(([kind, label]) => (
          <li key={kind} className={kind === current ? 'counter-cell counter-current' : 'counter-cell'}>
            <span className="counter-share">{Math.round(mix[kind] * 100)}%</span>
            <span className="counter-kind">{label}</span>
          </li>
        ))}
      </ul>
    </aside>
  );
}
