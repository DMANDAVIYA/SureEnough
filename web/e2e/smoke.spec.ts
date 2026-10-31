import { type APIRequestContext, expect, type Page, test } from '@playwright/test';
import { BADGE } from '../src/copy';
import type { Curated, CuratedLabel, Example, Manifest } from '../src/data/schema';
import { MS_PER_REPEAT } from '../src/lib/player';

const BUNDLE = '/SureEnough/data/mock/';
const START = new Date('2026-11-01T12:00:00Z').getTime();
const REPLAY_BUDGET_MS = 20 * MS_PER_REPEAT + 1000;

const pad = (n: number, width: number): string => String(n).padStart(width, '0');

const getJson = async <T>(request: APIRequestContext, path: string): Promise<T> =>
  (await request.get(`${BUNDLE}${path}`)).json() as Promise<T>;

async function curatedCase(request: APIRequestContext, label: CuratedLabel) {
  const manifest = await getJson<Manifest>(request, 'manifest.json');
  const curated = manifest.curated.find((c) => c.label === label) as Curated;
  const example = await getJson<Example>(request, `examples/sub-${pad(curated.subject, 2)}/${pad(curated.i, 3)}.json`);
  return { manifest, curated, example };
}

async function startCurated(page: Page, chip: string) {
  await page.clock.install({ time: START });
  await page.goto('/SureEnough/#/');
  await page.clock.pauseAt(START + 60_000);
  await page.getByRole('button', { name: chip, exact: true }).click();
  await expect(page.locator('.counter-now')).toHaveText('1');
}

async function replayMystery(page: Page) {
  await page.clock.install();
  await page.goto('/SureEnough/#/');
  await page.getByRole('button', { name: 'Pick a mystery' }).click();
  await expect(page.locator('.counter-now')).not.toHaveText('0');
  await page.clock.runFor(REPLAY_BUDGET_MS);
}

test('a mystery replays until the badge reports the stop', async ({ page }) => {
  await replayMystery(page);
  const badge = page.getByRole('region', { name: 'The promise' });
  await expect(badge).toContainText(BADGE);
  await expect(badge).toContainText(/Stopped after \d+ repeats?/);
});

test('an unsure case shows the exported set size at every repeat, even past the exported ranking', async ({
  page,
  request,
}) => {
  const { manifest, example } = await curatedCase(request, 'unsure');
  expect(example.set_size.some((size, t) => size > example.top[t].length)).toBe(true);
  await startCurated(page, 'Unsure');
  await page.clock.runFor(MS_PER_REPEAT / 2);
  for (let repeat = 1; repeat <= example.stopped_at; repeat++) {
    await expect(page.locator('.counter-now')).toHaveText(String(repeat));
    await expect(page.locator('.tiles-count')).toHaveText(String(example.set_size[repeat - 1]));
    await page.clock.runFor(MS_PER_REPEAT);
  }
  await expect(page.getByRole('region', { name: 'The promise' })).toContainText(`Stopped after ${manifest.K} repeats`);
});

test('a clean stop ends on the cutout of the one object left', async ({ page, request }) => {
  const { manifest, curated } = await curatedCase(request, 'clean stop');
  await startCurated(page, 'Clean stop');
  await page.clock.runFor(REPLAY_BUDGET_MS);
  const blend = page.getByRole('region', { name: 'What it thinks they saw' }).getByRole('img', { name: /^Blend of/ });
  await expect(blend.locator('img')).toHaveCount(1);
  await expect(blend.locator('img')).toHaveAttribute('src', new RegExp(`${manifest.concepts[curated.i].cutout}$`));
  await expect(page.getByText('The blend sharpens as repeats arrive.')).toHaveCount(0);
});

test('a hedged answer whose top pick was wrong still reveals the truth inside the answer', async ({ page }) => {
  await startCurated(page, 'Hedged, top pick wrong');
  await page.clock.runFor(REPLAY_BUDGET_MS);
  await expect(page.getByRole('region', { name: 'Reveal' })).toContainText("Inside SureEnough's final answer.");
});

test('a new random split reports the delivered error within a second', async ({ page }) => {
  await replayMystery(page);
  const button = page.getByRole('button', { name: 'New random split' });
  await expect(button).toBeEnabled();
  await button.click();
  await expect(page.getByTestId('delivered')).toHaveText(/^\d+\.\d in 10$/, { timeout: 1000 });
});

test('the results deep link renders the promise chart', async ({ page }) => {
  await page.goto('/SureEnough/#/results');
  const chart = page.getByRole('img', { name: 'Promised vs delivered error per person' });
  await expect(chart).toBeVisible();
  await expect(chart.locator('circle')).not.toHaveCount(0);
});

test('changing the strictness recalibrates without freezing the page', async ({ page }) => {
  await page.goto('/SureEnough/#/results');
  const chart = page.getByRole('img', { name: 'Promised vs delivered error per person' });
  await expect(chart).toBeVisible();
  await page.evaluate(() => {
    const durations: number[] = [];
    new PerformanceObserver((list) => durations.push(...list.getEntries().map((entry) => entry.duration))).observe({
      type: 'longtask',
    });
    Object.assign(window, { longTasks: durations });
  });
  await page.getByRole('radio', { name: /at most 3 in 10/ }).check();
  await expect(page.getByText('Recalibrating…')).toHaveCount(0);
  await expect(chart).toBeVisible();
  const longest = await page.evaluate(() => Math.max(0, ...(window as unknown as { longTasks: number[] }).longTasks));
  expect(longest).toBeLessThan(100);
});

test('a view that cannot read the export shows an alert instead of going blank', async ({ page }) => {
  await page.route('**/manifest.json', async (route) => {
    const response = await route.fetch();
    const manifest = await response.json();
    delete manifest.results.per_subject;
    await route.fulfill({ response, json: manifest });
  });
  for (const view of ['#/', '#/results']) {
    await page.goto(`/SureEnough/${view}`);
    await expect(page.getByRole('alert')).toContainText('could not be shown');
  }
  await page.getByRole('link', { name: 'How it works' }).click();
  await expect(page.getByRole('heading', { name: 'How SureEnough keeps its promise' })).toBeVisible();
});

test.describe('on a 390 px phone', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  const scrollWidth = (page: Page) => page.evaluate(() => document.documentElement.scrollWidth);

  test('no view scrolls sideways, even with the replay finished and the provenance open', async ({ page }) => {
    await replayMystery(page);
    await page.getByText('Show the Muse Spark and SAM 3.1 calls behind this example').click();
    await page.getByRole('button', { name: 'New random split' }).click();
    expect(await scrollWidth(page)).toBeLessThanOrEqual(390);
    for (const route of ['#/results', '#/how']) {
      await page.goto(`/SureEnough/${route}`);
      await expect(page.getByRole('heading', { level: 2 }).first()).toBeVisible();
      expect(await scrollWidth(page)).toBeLessThanOrEqual(390);
    }
  });
});
