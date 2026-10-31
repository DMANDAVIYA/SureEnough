import { NOT_COVERED } from '../copy';
import type { Example } from '../data/schema';

interface Props {
  example: Example | null;
  stopped: boolean;
  narratedSubjects: number[];
}

export default function Narration({ example, stopped, narratedSubjects }: Props) {
  const done = stopped && example !== null;
  return (
    <section className="panel words" aria-label="In words">
      <h2>In words</h2>
      <p className={done ? 'claim' : 'claim claim-pending'}>{done ? example.claim.text : 'No answer yet.'}</p>
      {done && example.narration && (
        <blockquote className="narration">
          <p>{example.narration}</p>
          <footer>Muse Spark, restating the claim</footer>
        </blockquote>
      )}
      {done && !example.narration && (
        <p className="panel-note">Muse Spark narration covers people {narratedSubjects.join(' and ')} only.</p>
      )}
      {done && (
        <p className="pe-guess">
          Perception Encoder's open-vocabulary guess over all 1,854 THINGS nouns: <em>{example.pe_guess}</em>{' '}
          <span className="tag">{NOT_COVERED}</span>
        </p>
      )}
    </section>
  );
}
