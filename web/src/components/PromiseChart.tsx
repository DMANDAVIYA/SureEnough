import { inTen } from '../copy';

export interface RiskRow {
  label: string;
  sure: number;
  naive: number;
}

const WIDTH = 360;
const PLOT_LEFT = 84;
const PLOT_RIGHT = 340;
const TOP = 30;
const ROW = 34;
const AXIS = 42;
const MARK = 6;

const scaleMax = (rows: RiskRow[], alpha: number): number =>
  Math.min(1, Math.max(0.5, Math.ceil(Math.max(alpha, ...rows.flatMap((r) => [r.sure, r.naive])) * 10 + 1) / 10));

export function ChartLegend() {
  return (
    <ul className="chart-legend">
      <li>
        <svg width="14" height="14" viewBox="-7 -7 14 14" aria-hidden="true">
          <circle r={MARK} className="mark-sure" />
        </svg>
        SureEnough
      </li>
      <li>
        <svg width="14" height="14" viewBox="-7 -7 14 14" aria-hidden="true">
          <rect x={-5} y={-5} width={10} height={10} transform="rotate(45)" className="mark-naive" />
        </svg>
        Naive repeated look
      </li>
      <li>
        <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
          <line x1="7" x2="7" y1="0" y2="14" className="promise-line" />
        </svg>
        Promise
      </li>
    </ul>
  );
}

export default function PromiseChart({ rows, alpha, title }: { rows: RiskRow[]; alpha: number; title: string }) {
  const max = scaleMax(rows, alpha);
  const x = (v: number): number => PLOT_LEFT + (v / max) * (PLOT_RIGHT - PLOT_LEFT);
  const bottom = TOP + rows.length * ROW;
  const ticks = Array.from({ length: Math.round(max * 10) + 1 }, (_, k) => k);
  return (
    <svg className="promise-chart" viewBox={`0 0 ${WIDTH} ${bottom + AXIS}`} role="img" aria-label={title}>
      <rect x={x(alpha)} y={TOP - 8} width={PLOT_RIGHT - x(alpha)} height={bottom - TOP + 8} className="zone-over" />
      <text x={PLOT_RIGHT} y={TOP - 14} className="chart-note" textAnchor="end">
        over the promise
      </text>
      {ticks.map((k) => (
        <g key={k}>
          <line x1={x(k / 10)} x2={x(k / 10)} y1={TOP - 8} y2={bottom} className="gridline" />
          <text x={x(k / 10)} y={bottom + 16} className="chart-tick" textAnchor="middle">
            {k}
          </text>
        </g>
      ))}
      <text x={(PLOT_LEFT + PLOT_RIGHT) / 2} y={bottom + 34} className="chart-note" textAnchor="middle">
        misses per 10 answers
      </text>
      <line x1={x(alpha)} x2={x(alpha)} y1={TOP - 22} y2={bottom} className="promise-line" />
      <text x={x(alpha) - 6} y={TOP - 14} className="chart-note" textAnchor="end">
        promise
      </text>
      {rows.map((row, k) => {
        const y = TOP + k * ROW + ROW / 2;
        return (
          <g key={row.label}>
            <text x={0} y={y + 4} className="chart-label">
              {row.label}
            </text>
            <line x1={PLOT_LEFT} x2={PLOT_RIGHT} y1={y} y2={y} className="row-track" />
            <rect x={-5} y={-5} width={10} height={10} transform={`translate(${x(row.naive)} ${y}) rotate(45)`} className="mark-naive">
              <title>{`${row.label}, naive repeated look: ${inTen(row.naive)}`}</title>
            </rect>
            <circle cx={x(row.sure)} cy={y} r={MARK} className="mark-sure">
              <title>{`${row.label}, SureEnough: ${inTen(row.sure)}`}</title>
            </circle>
          </g>
        );
      })}
    </svg>
  );
}
