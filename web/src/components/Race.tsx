import { inTen } from '../copy';
import type { Findings, RaceLane, RaceLanes } from '../lib/results';

interface Props {
  race: RaceLanes;
  h1: Findings['h1'];
  budget: number;
}

function Lane({ name, note, lane, budget, own }: { name: string; note: string; lane: RaceLane; budget: number; own: boolean }) {
  return (
    <li className={own ? 'lane lane-sure' : 'lane'}>
      <p className="lane-name">
        <strong>{name}</strong> {note}
      </p>
      <div className="lane-track">
        <span className="lane-rail">
          <span className="lane-bar" style={{ width: `${(lane.repeats / budget) * 100}%` }} />
        </span>
        <span className="lane-value">{lane.repeats.toFixed(1)} repeats</span>
      </div>
      <p className="lane-detail">
        Wrong {inTen(lane.wrong)}, answer size {lane.size.toFixed(1)} objects
      </p>
    </li>
  );
}

export default function Race({ race, h1, budget }: Props) {
  if (!h1.holds && !h1.parity) return null;
  return (
    <section className="panel race" aria-label="Race">
      <h2>{h1.holds ? 'Same promise, fewer repeats' : 'Same promise, about as many repeats'}</h2>
      <p className="panel-note">Both rules are calibrated to at most 2 in 10 for this person. Averages over the evaluation half.</p>
      <ul className="lanes">
        <Lane name="SureEnough" note="stops when it has seen enough" lane={race.sure} budget={budget} own />
        <Lane name="Best fixed rule" note={`always listens for ${race.fixed.repeats} repeats`} lane={race.fixed} budget={budget} own={false} />
      </ul>
    </section>
  );
}
