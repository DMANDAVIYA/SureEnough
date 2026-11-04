import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import type { Example } from '../data/schema';
import Badge from './Badge';
import Footer from './Footer';
import Header from './Header';
import Narration from './Narration';
import Reveal from './Reveal';

it('header shows the hook and the verbatim replay label', () => {
  const html = renderToStaticMarkup(<Header />);
  expect(html).toContain('Brain decoders always guess.');
  expect(html).toContain(
    'Replayed public lab EEG (THINGS-EEG2). Not live. Each repeat is a 100 ms flash in a rapid sequence.',
  );
});

it('footer shows the verbatim credit', () => {
  expect(renderToStaticMarkup(<Footer />)).toContain(
    'EEG: THINGS-EEG2 (Gifford et al. 2022), CC-BY 4.0 · Object photos: THINGSplus, CC0.',
  );
});

it('badge shows the verbatim promise and the stop in repeats', () => {
  const html = renderToStaticMarkup(<Badge stoppedAt={7} />);
  expect(html).toContain(
    'Promise: for this person, the true object is outside SureEnough&#x27;s final answer at most 2 in 10 times (checked with 90% confidence).',
  );
  expect(html).toContain('Stopped after 7 repeats');
  expect(html).not.toMatch(/second|probably|session/i);
});

it('reveal shows the verbatim licence caption', () => {
  const concept = { i: 3, name: 'wolf', things_id: 'wolf', tile: 'img/tiles/wolf.svg', cutout: 'img/cutouts/wolf.svg', category: 'animal' };
  expect(renderToStaticMarkup(<Reveal concept={concept} hit />)).toContain(
    'A different photo of the same thing. The real stimulus can&#x27;t be shown (licence).',
  );
});

const spoken = (narration: string | null, narration_rejected?: boolean) =>
  ({ claim: { kind: 'list', text: "It's one of these 14.", category: null }, narration, narration_rejected, pe_guess: 'wheel' }) as unknown as Example;

it('narration credits Muse Spark only for a sentence that passed the faithfulness check', () => {
  const kept = renderToStaticMarkup(<Narration example={spoken('You saw a bike or a wheel.')} stopped narratedSubjects={[6]} />);
  expect(kept).toContain('You saw a bike or a wheel.');
  expect(kept).toContain('Muse Spark, restating the claim');
  const rejected = renderToStaticMarkup(<Narration example={spoken(null, true)} stopped narratedSubjects={[6]} />);
  expect(rejected).not.toContain('Muse Spark, restating the claim');
  expect(rejected).not.toContain('covers people');
  expect(rejected).toContain('sentence did not pass the faithfulness check');
});
