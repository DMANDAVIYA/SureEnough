import type { ApiExchange, Example } from '../data/schema';

function Exchange({ title, exchange }: { title: string; exchange: ApiExchange }) {
  return (
    <section className="exchange">
      <h3>{title}</h3>
      <p className="exchange-label">Request</p>
      <pre>{JSON.stringify(exchange.request, null, 2)}</pre>
      <p className="exchange-label">Response</p>
      <pre>{JSON.stringify(exchange.response, null, 2)}</pre>
    </section>
  );
}

export default function Provenance({ provenance }: { provenance: Example['provenance'] }) {
  if (!provenance) return null;
  return (
    <details className="panel provenance">
      <summary>Show the Muse Spark and SAM 3.1 calls behind this example</summary>
      {provenance.spark && <Exchange title="Muse Spark 1.3, restating the claim" exchange={provenance.spark} />}
      {provenance.sam && <Exchange title="SAM 3.1, cutting out the leading candidate" exchange={provenance.sam} />}
    </details>
  );
}
