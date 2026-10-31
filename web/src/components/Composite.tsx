import type { Concept } from '../data/schema';
import { stackedOpacities } from '../lib/blend';
import { dataUrl } from '../lib/data';

interface Layer {
  concept: Concept;
  weight: number;
}

const listOf = (names: string[]): string =>
  names.length < 2 ? names.join('') : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;

function Blend({ layers }: { layers: Layer[] }) {
  const opacities = stackedOpacities(layers.map((layer) => layer.weight));
  return (
    <div className="blend" role="img" aria-label={`Blend of ${listOf(layers.map((layer) => layer.concept.name))}`}>
      {layers.map(({ concept }, k) => (
        <img key={concept.i} src={dataUrl(concept.cutout)} alt="" style={{ opacity: opacities[k] }} />
      ))}
    </div>
  );
}

interface Props {
  layers: Layer[];
  impression: string | null;
}

export default function Composite({ layers, impression }: Props) {
  return (
    <section className="panel seen" aria-label="What it thinks they saw">
      <h2>What it thinks they saw</h2>
      <div className="frames">
        <figure className="frame">
          {layers.length > 0 ? <Blend layers={layers} /> : <div className="frame-empty">The blend sharpens as repeats arrive.</div>}
          <figcaption>SAM 3.1 cutouts of the leading candidates, blended by likelihood</figcaption>
        </figure>
        {impression && (
          <figure className="frame frame-impression">
            <img src={dataUrl(impression)} alt="Artist's impression of the final answer" />
            <figcaption>Muse Image impression of the final answer</figcaption>
          </figure>
        )}
      </div>
    </section>
  );
}
