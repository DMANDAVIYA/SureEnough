const STAGES = [
  { name: 'Perception Encoder', role: 'The space the EEG is decoded into, and the open-vocabulary guess.', optional: false },
  { name: 'SureEnough calibration', role: 'Learn then Test decides when to stop and how wide the answer must be.', optional: false },
  { name: 'SAM 3.1 composites', role: 'Cuts each candidate out of its CC0 photo, blended by likelihood.', optional: false },
  { name: 'Muse Spark words', role: 'Restates the calibrated claim in one sentence, checked for faithfulness.', optional: false },
  { name: 'Muse Image', role: "An artist's impression of the final answer, only where the model is available.", optional: true },
];

export default function MetaStack() {
  return (
    <figure className="meta-stack">
      <ol>
        {STAGES.map((stage) => (
          <li key={stage.name} className={stage.optional ? 'meta-step-optional' : undefined}>
            <strong>{stage.name}</strong>
            <span>{stage.role}</span>
          </li>
        ))}
      </ol>
      <figcaption>How the Meta models fit together, from brain signal to words and pictures.</figcaption>
    </figure>
  );
}
