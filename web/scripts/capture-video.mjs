import { execFileSync } from 'node:child_process';
import { mkdirSync, readdirSync, rmSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { chromium } from '@playwright/test';

process.env.PLAYWRIGHT_BROWSERS_PATH ??= '0';

const BASE = process.argv[2] ?? 'http://localhost:4175/SureEnough/';
const OUT = resolve(process.argv[3] ?? '../video/out');
const RAW = join(OUT, 'raw');
const SIZE = { width: 1920, height: 1080 };
const MS_PER_REPEAT = 600;
const BUDGET = 20;

const OVERLAY = `
  const card = document.createElement('div');
  card.id = 'cap-card';
  card.style.cssText = 'position:fixed;inset:0;z-index:99999;display:none;flex-direction:column;align-items:center;justify-content:center;gap:28px;background:#0d1117;color:#f0f3f6;font:600 64px/1.2 system-ui;text-align:center;padding:0 160px';
  const line = document.createElement('div');
  line.id = 'cap-line';
  line.style.cssText = 'position:fixed;left:50%;top:28px;transform:translateX(-50%);z-index:99998;max-width:1500px;display:none;padding:18px 30px;border-radius:14px;background:rgba(13,17,23,.88);color:#fff;font:500 34px/1.35 system-ui;text-align:center';
  document.body.append(card, line);
`;

const wait = (ms) => new Promise((done) => setTimeout(done, ms));

async function card(page, title, subtitle = '', ms = 4000) {
  await page.evaluate(
    ([t, s]) => {
      const el = document.getElementById('cap-card');
      el.innerHTML = '';
      const a = document.createElement('div');
      a.textContent = t;
      el.append(a);
      if (s) {
        const b = document.createElement('div');
        b.textContent = s;
        b.style.cssText = 'font:400 36px/1.4 system-ui;color:#9fb0c3;max-width:1300px';
        el.append(b);
      }
      el.style.display = 'flex';
    },
    [title, subtitle],
  );
  await wait(ms);
}

async function clearCard(page) {
  await page.evaluate(() => (document.getElementById('cap-card').style.display = 'none'));
}

async function caption(page, text) {
  await page.evaluate((t) => {
    const el = document.getElementById('cap-line');
    el.textContent = t;
    el.style.display = t ? 'block' : 'none';
  }, text);
}

async function scrollTo(page, name) {
  await page.getByRole('region', { name }).first().evaluate((el) => el.scrollIntoView({ behavior: 'smooth', block: 'center' }));
  await wait(1200);
}

async function replay(page, chip, linger) {
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'smooth' }));
  await page.getByRole('button', { name: chip, exact: true }).first().click();
  await page.locator('.reveal').waitFor({ state: 'detached' });
  await page.locator('.reveal').waitFor({ state: 'attached', timeout: BUDGET * MS_PER_REPEAT + 5000 });
  await wait(linger);
}

async function main() {
  rmSync(OUT, { recursive: true, force: true });
  mkdirSync(RAW, { recursive: true });
  const browser = await chromium.launch();
  const context = await browser.newContext({ viewport: SIZE, recordVideo: { dir: RAW, size: SIZE }, colorScheme: 'light' });
  const page = await context.newPage();
  const started = Date.now();
  await page.goto(BASE);
  await page.getByRole('button', { name: 'Pick a mystery' }).waitFor();
  await page.evaluate(OVERLAY);
  const lead = (Date.now() - started) / 1000;

  await card(page, 'Brain decoders always guess.', '', 3500);
  await card(page, 'SureEnough knows when it has seen enough,', 'says "not sure" when it isn\'t, and keeps its promise about how often it\'s wrong.', 5500);
  await clearCard(page);
  await caption(page, 'Replayed public lab EEG (THINGS-EEG2, 17 channels). Not live. Each repeat is a 100 ms flash.');
  await wait(5500);

  await caption(page, 'Each repeat narrows 200 objects to a shortlist. It stops as soon as it is sure enough.');
  await replay(page, 'Clean stop', 2500);
  await scrollTo(page, 'What it thinks they saw');
  await wait(3500);

  await caption(page, "When it isn't sure, it keeps listening and answers with a list instead of a confident wrong guess.");
  await replay(page, 'Unsure', 4000);

  await caption(page, 'Here the top pick is wrong, but the answer still contains the truth.');
  await replay(page, 'Hedged, top pick wrong', 2500);
  await caption(page, 'Muse Spark says the answer in one checked sentence. SAM 3.1 cutouts and Muse Image show it.');
  await scrollTo(page, 'What it thinks they saw');
  await wait(4500);
  await scrollTo(page, 'Reveal');
  await wait(3500);

  await caption(page, 'The promise: the true object is outside the answer at most 2 in 10 times, checked on images it never saw.');
  await scrollTo(page, 'Live recalibration');
  for (let k = 0; k < 3; k += 1) {
    await page.getByRole('button', { name: 'New random split' }).click();
    await wait(3000);
  }

  await page.getByRole('link', { name: 'Results' }).click();
  await caption(page, '10 people, 1,000 random checks each: the promise never broke.');
  await wait(5000);
  await scrollTo(page, 'Findings');
  await caption(page, 'Honest catch: with today\'s EEG decoders, being sure still takes about 18 of 20 repeats.');
  await wait(6500);

  await page.getByRole('link', { name: 'How it works' }).click();
  await scrollTo(page, 'Meta stack');
  await caption(page, 'Built on Meta Perception Encoder, SAM 3.1, Muse Spark and Muse Image.');
  await wait(6000);
  await caption(page, '');
  await card(page, 'SureEnough', 'dmandaviya.github.io/SureEnough  ·  github.com/DMANDAVIYA/SureEnough', 5000);

  await context.close();
  await browser.close();
  const webm = join(RAW, readdirSync(RAW).find((f) => f.endsWith('.webm')));
  const mp4 = join(OUT, 'sureenough.mp4');
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-ss', lead.toFixed(2), '-i', webm, '-r', '30', '-c:v', 'libx264', '-crf', '18', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', mp4]);
  const seconds = execFileSync('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', mp4]).toString().trim();
  console.log(`${mp4} ${Number(seconds).toFixed(1)}s`);
}

await main();
