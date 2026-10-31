const WIDTH = 500;
const HEIGHT = 170;
const MIDLINE = HEIGHT / 2;
const AMPLITUDE = 72;
const ONSET_X = WIDTH / 5;

const points = (trace: number[]): string =>
  trace.map((value, k) => `${((k / (trace.length - 1)) * WIDTH).toFixed(1)},${(MIDLINE - (value / 127) * AMPLITUDE).toFixed(1)}`).join(' ');

interface Props {
  first: number[] | null;
  current: number[] | null;
  repeat: number;
  budget: number;
  stopped: boolean;
}

export default function BrainSignal({ first, current, repeat, budget, stopped }: Props) {
  return (
    <section className="panel signal" aria-label="Brain signal">
      <h2>Brain signal</h2>
      <p className="panel-note">The averaged response at the back of the head, from lab EEG (17 channels). Noise cancels as repeats add up.</p>
      <figure className="paper">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={current ? `Average of ${repeat} repeats` : 'No replay yet'}>
          <defs>
            <pattern id="paper-minor" width="20" height="17" patternUnits="userSpaceOnUse">
              <path d="M20 0H0V17" className="paper-line" />
            </pattern>
            <pattern id="paper-major" width="100" height="85" patternUnits="userSpaceOnUse">
              <rect width="100" height="85" fill="url(#paper-minor)" />
              <path d="M100 0H0V85" className="paper-line-major" />
            </pattern>
          </defs>
          <rect width={WIDTH} height={HEIGHT} fill="url(#paper-major)" />
          <line x1={ONSET_X} x2={ONSET_X} y1="0" y2={HEIGHT} className="onset" />
          {first && repeat > 1 && <polyline points={points(first)} className="trace trace-single" />}
          <polyline points={current ? points(current) : `0,${MIDLINE} ${WIDTH},${MIDLINE}`} className="trace" />
        </svg>
        <figcaption className="paper-axis">
          <span>−200</span>
          <span className="paper-onset">flash</span>
          <span>800 ms</span>
        </figcaption>
      </figure>
      <div className="signal-foot">
        <p className="counter">
          <span className="counter-now">{repeat}</span> of {budget}
          <span className="counter-unit">repeats averaged</span>
        </p>
        <ol className="track" aria-hidden="true">
          {Array.from({ length: budget }, (_, k) => (
            <li key={k} className={k < repeat ? (stopped && k === repeat - 1 ? 'tick tick-stop' : 'tick tick-done') : 'tick'} />
          ))}
        </ol>
      </div>
      {first && repeat > 1 && (
        <p className="legend">
          <span className="swatch swatch-single" /> one repeat <span className="swatch" /> average of {repeat}
        </p>
      )}
    </section>
  );
}
