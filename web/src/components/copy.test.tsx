import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import Footer from './Footer';
import Header from './Header';

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
