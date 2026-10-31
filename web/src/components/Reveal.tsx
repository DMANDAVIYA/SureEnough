import { REVEAL_CAPTION } from '../copy';
import type { Concept } from '../data/schema';
import { dataUrl } from '../lib/data';

export default function Reveal({ concept, hit }: { concept: Concept; hit: boolean }) {
  return (
    <section className="panel reveal" aria-label="Reveal">
      <h2>What they were looking at</h2>
      <figure className="reveal-figure">
        <img src={dataUrl(concept.tile)} alt={concept.name} width={144} height={144} />
        <figcaption>
          <span className="reveal-name">{concept.name}</span>
          <span className="reveal-caption">{REVEAL_CAPTION}</span>
        </figcaption>
      </figure>
      <p className={hit ? 'verdict' : 'verdict verdict-miss'}>
        {hit
          ? "Inside SureEnough's final answer."
          : "Outside SureEnough's final answer. This is one of the misses the promise allows: at most 2 in 10."}
      </p>
    </section>
  );
}
