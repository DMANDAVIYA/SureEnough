import { BADGE, stoppedAfter } from '../copy';
import Pips from './Pips';

export default function Badge({ stoppedAt }: { stoppedAt?: number | null }) {
  return (
    <section className="badge" aria-label="The promise" title={BADGE}>
      <Pips total={10} open={2} />
      <div className="badge-body">
        <p className="badge-text">{BADGE}</p>
        {stoppedAt !== undefined && (
          <p className="badge-stop" role="status">
            {stoppedAt === null ? 'Still adding repeats' : stoppedAfter(stoppedAt)}
          </p>
        )}
      </div>
    </section>
  );
}
