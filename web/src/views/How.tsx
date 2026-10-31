import MetaStack from '../components/MetaStack';

export default function How() {
  return (
    <div className="how">
      <article className="panel">
        <h2>How SureEnough keeps its promise</h2>

        <h3>One dial</h3>
        <p>
          After every repeat, SureEnough lines up the 200 objects from most to least likely and keeps the smallest group
          whose combined likelihood reaches a dial setting, λ. It stops as soon as that group has 3 objects or fewer, or
          after 20 repeats. Turning the dial down gives smaller answers and earlier stops, and more misses.
        </p>

        <h3>Learn then Test</h3>
        <p>
          The dial is set on 100 calibration images. Settings are tried in a fixed order chosen before any results, from
          the safest (λ = 1: keep all 200 and never stop early) toward bolder ones. At each setting we test the claim
          that its error is above the promise. We keep going while every test rejects that claim, and deploy the last
          setting that passed. Because the order is fixed, no correction for testing many settings is needed.
        </p>

        <h3>α and δ</h3>
        <p>
          α = 0.2 is the error the promise allows: the true object may be outside the final answer at most 2 in 10 times.
          δ = 0.1 is the chance that the 100 calibration images were unluckily easy. The promise holds with 90% confidence
          over that draw: the 90% is about the calibration, not about any single answer.
        </p>

        <h3>Hoeffding–Bentkus</h3>
        <p>
          Each test turns the error seen on the calibration images into a p-value. The Hoeffding–Bentkus bound takes the
          smaller of two classic tail bounds, Hoeffding's and Bentkus's binomial one. Both hold for any error between 0 and
          1 without assuming a shape for the errors, and together they stay tight with only 100 images.
        </p>

        <h3>Why checking after every repeat fails</h3>
        <p>
          The naive way calibrates a separate threshold for each repeat and stops at the first small answer. Each
          threshold is valid on its own, but stopping at the first lucky look breaks the promise. The Results tab shows by
          how much.
        </p>

        <h3>What the promise covers</h3>
        <p>
          A random image from this person's 200-image gallery, with these recordings replayed in a random order. New
          images or galleries are an extra assumption. The open-vocabulary guess and the pictures are not covered.
        </p>
      </article>
      <section className="panel" aria-label="Meta stack">
        <h2>The Meta stack</h2>
        <MetaStack />
      </section>
    </div>
  );
}
