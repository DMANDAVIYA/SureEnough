import { useEffect, useMemo, useState } from 'react';
import Badge from '../components/Badge';
import BrainSignal from '../components/BrainSignal';
import Composite from '../components/Composite';
import Narration from '../components/Narration';
import OutcomeCounters from '../components/OutcomeCounters';
import Provenance from '../components/Provenance';
import Race from '../components/Race';
import Recalibrate from '../components/Recalibrate';
import Reveal from '../components/Reveal';
import TileGrid from '../components/TileGrid';
import { useManifest } from '../data/manifest';
import type { Curated } from '../data/schema';
import { ready, useResource } from '../data/useResource';
import { loadExample, loadTable } from '../lib/data';
import { createPlayer, play, type Player } from '../lib/player';
import { findingsOf, meanOutcomeMix, raceOf, subjectSummary } from '../lib/results';
import { mulberry32 } from '../lib/rng';
import { pickMystery } from '../lib/splits';

interface Selection {
  subject: number;
  i: number;
  label: string;
}

const capitalise = (text: string): string => text[0].toUpperCase() + text.slice(1);

export default function Demo() {
  const manifest = useManifest();
  const random = useMemo(() => mulberry32(Date.now()), []);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [run, setRun] = useState(0);
  const [tick, setTick] = useState<{ player: Player; repeat: number } | null>(null);

  const loadedExample = useResource(selection && `${selection.subject}/${selection.i}`, () =>
    loadExample(selection!.subject, selection!.i),
  );
  const loadedTable = useResource(selection && `table/${selection.subject}`, () => loadTable(selection!.subject));
  const example = ready(loadedExample);

  const player = useMemo(() => (example ? createPlayer(example) : null), [example]);

  useEffect(() => (player ? play(player.stopAt, (repeat) => setTick({ player, repeat })) : undefined), [player, run]);

  const frame = player && tick?.player === player ? player.step(tick.repeat) : null;
  const stopped = frame?.stopped ?? false;
  const members = frame?.setMembers ? new Set(frame.setMembers) : null;
  const layers = frame ? frame.blend.map(({ i, weight }) => ({ concept: manifest.concepts[i], weight })) : [];
  const truth = selection ? manifest.concepts[selection.i] : null;
  const findings = useMemo(() => findingsOf(manifest.results), [manifest]);
  const summary = selection ? subjectSummary(manifest.results, selection.subject) : undefined;
  const partialMuse = manifest.demo_subjects.length < manifest.subjects.length;

  const pick = () => setSelection({ ...pickMystery(manifest, random), label: 'A mystery' });
  const choose = (curated: Curated) => setSelection({ subject: curated.subject, i: curated.i, label: capitalise(curated.label) });

  return (
    <div className="demo">
      <div className="controls">
        <button type="button" className="button button-primary" onClick={pick}>
          Pick a mystery
        </button>
        <div className="curated" role="group" aria-labelledby="curated-label">
          <span id="curated-label" className="curated-label">
            Curated examples
          </span>
          {manifest.curated.map((curated) => (
            <button
              key={`${curated.subject}-${curated.i}`}
              type="button"
              className="chip"
              aria-pressed={selection?.subject === curated.subject && selection.i === curated.i}
              onClick={() => choose(curated)}
            >
              {capitalise(curated.label)}
            </button>
          ))}
        </div>
      </div>
      {partialMuse && (
        <p className="panel-note">
          Replays cover all {manifest.subjects.length} people. Muse Spark words and Muse Image pictures cover people{' '}
          {manifest.demo_subjects.join(' and ')}.
        </p>
      )}
      {selection && (
        <p className="now-playing">
          {selection.label}, person {selection.subject}
          {stopped && (
            <button type="button" className="link-button" onClick={() => setRun((n) => n + 1)}>
              Watch again
            </button>
          )}
        </p>
      )}
      {loadedExample instanceof Error && (
        <p className="status" role="alert">
          This replay did not load ({loadedExample.message}). Pick another mystery.
        </p>
      )}
      <div className="stage">
        <BrainSignal
          first={example?.eeg[0] ?? null}
          current={frame && example ? example.eeg[frame.repeat - 1] : null}
          repeat={frame?.repeat ?? 0}
          budget={manifest.K}
          stopped={stopped}
        />
        <TileGrid concepts={manifest.concepts} setSize={frame?.setSize ?? null} members={members} />
        <Composite layers={layers} impression={stopped && manifest.muse_image ? (example?.impression ?? null) : null} />
        <Narration example={example} stopped={stopped} narratedSubjects={manifest.demo_subjects} />
        <Badge stoppedAt={selection ? (frame && stopped ? frame.repeat : null) : undefined} />
        {stopped && truth && example && <Reveal concept={truth} hit={example.truth_in_set} />}
        {selection && (
          <Recalibrate
            key={selection.subject}
            table={ready(loadedTable)}
            alpha={manifest.alpha}
            delta={manifest.delta}
            images={manifest.concepts.length}
          />
        )}
        {summary && <Race race={raceOf(summary)} h1={findings.h1} budget={manifest.K} />}
        {stopped && example && <Provenance provenance={example.provenance} />}
      </div>
      <OutcomeCounters
        title={summary ? `How person ${selection?.subject}'s answers ended, evaluation half` : 'How answers ended, all people'}
        mix={summary?.outcome_mix ?? meanOutcomeMix(manifest.results)}
        current={stopped && example ? example.claim.kind : null}
      />
    </div>
  );
}
