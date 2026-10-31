import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'data', 'mock');
const SUBJECTS = [1, 2];
const SUBJECT_GAIN = { 1: 1.25, 2: 1.0 };
const K = 20;
const R = 10;
const M = 3;
const ALPHA = 0.2;
const DELTA = 0.1;
const U16 = 65535;
const TOP = 30;
const LAM = Array.from({ length: 500 }, (_, j) => 1 - j / 500);
const MOCK_RISK_CAP = 0.1;
const MAX_CATEGORY_SIZE = 50;
const FIXED_K = [1, 2, 4, 8, 16, 20];
const BETA = 1.6;
const SIGNAL = 1.5;
const NOISE = 1.0;
const CONFUSED_SHARE = 0.12;
const IMPRESSION_MEMBERS = 6;
const MOCK_H1_P = 0.004;
const MOCK_CP_UPPER = 0.087;
const SPARK_SYSTEM =
  'Rewrite the claim about which object a person saw as one short, plain English sentence. ' +
  'Use only the object or category names you are given and add no other objects or categories. ' +
  'Never hedge with words like probably, likely or maybe, and give no percentages. ' +
  'Do not mention brains, EEG or models.';

const CATEGORIES = {
  animal: ['wolf', 'husky', 'coyote', 'fox', 'bear', 'deer', 'moose', 'rabbit', 'squirrel', 'beaver', 'otter', 'seal',
    'horse', 'donkey', 'zebra', 'camel', 'goat', 'sheep', 'pig', 'cow', 'lion', 'tiger', 'leopard', 'monkey'],
  bird: ['owl', 'eagle', 'parrot', 'penguin', 'duck', 'swan', 'pigeon', 'flamingo'],
  insect: ['bee', 'butterfly', 'beetle', 'ant', 'dragonfly', 'ladybug'],
  food: ['apple', 'banana', 'pear', 'peach', 'plum', 'cherry', 'grape', 'lemon', 'orange', 'lime', 'mango', 'pineapple',
    'kiwi', 'strawberry', 'watermelon', 'coconut', 'carrot', 'potato', 'onion', 'garlic', 'tomato', 'cucumber', 'pepper',
    'broccoli', 'cabbage', 'lettuce', 'corn', 'pumpkin', 'mushroom', 'bread', 'bagel', 'croissant', 'pretzel', 'muffin',
    'cake', 'cookie', 'donut', 'pie', 'pizza', 'sandwich', 'hamburger', 'hot dog', 'taco', 'burrito', 'sushi', 'noodle',
    'rice', 'egg', 'cheese', 'butter', 'yogurt', 'popcorn', 'chocolate', 'candy'],
  vehicle: ['car', 'bus', 'truck', 'tractor', 'bicycle', 'motorcycle', 'scooter', 'train', 'tram', 'airplane',
    'helicopter', 'boat', 'canoe', 'submarine'],
  tool: ['hammer', 'screwdriver', 'wrench', 'saw', 'drill', 'shovel', 'rake', 'axe', 'pliers', 'chisel', 'ladder',
    'tape measure', 'paintbrush', 'flashlight', 'crowbar', 'trowel'],
  clothing: ['shirt', 'sweater', 'jacket', 'coat', 'dress', 'skirt', 'jeans', 'scarf', 'glove', 'sock', 'boot',
    'sandal', 'hat', 'tie'],
  furniture: ['chair', 'table', 'sofa', 'bed', 'desk', 'shelf', 'stool', 'bench', 'cabinet', 'dresser', 'wardrobe',
    'hammock'],
  'musical instrument': ['guitar', 'violin', 'piano', 'drum', 'trumpet', 'flute', 'harp', 'accordion', 'banjo',
    'saxophone'],
  'sports equipment': ['football', 'basketball', 'tennis racket', 'baseball bat', 'skateboard', 'ski', 'surfboard',
    'golf club', 'hockey stick', 'dumbbell'],
  plant: ['cactus', 'fern', 'tulip', 'rose', 'sunflower', 'palm tree', 'bamboo', 'ivy'],
};
const UNCATEGORISED = ['lamp', 'clock', 'mirror', 'umbrella', 'key', 'lock', 'candle', 'bucket', 'basket', 'bottle',
  'cup', 'kettle', 'teapot', 'pan', 'spoon', 'fork', 'knife', 'bowl', 'vase', 'balloon', 'kite', 'globe', 'compass',
  'telescope'];
const MASS_SUFFIXES = ['clothing', 'equipment', 'food', 'furniture'];

const SHAPES = [
  '<circle cx="50" cy="46" r="26"/>',
  '<rect x="26" y="22" width="48" height="48" rx="8"/>',
  '<polygon points="50,18 80,72 20,72"/>',
  '<polygon points="50,16 80,46 50,76 20,46"/>',
  '<polygon points="50,18 76,33 76,61 50,76 24,61 24,33"/>',
  '<path d="M22 70 Q50 6 78 70 Z"/>',
];

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const gaussian = (random) => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());

function shuffled(values, random) {
  const out = [...values];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
const round3 = (x) => Math.round(x * 1000) / 1000;
const pad = (n, width) => String(n).padStart(width, '0');

const concepts = Object.entries(CATEGORIES)
  .flatMap(([category, names]) => names.map((name) => ({ name, category })))
  .concat(UNCATEGORISED.map((name) => ({ name, category: null })))
  .sort((a, b) => a.name.localeCompare(b.name))
  .map(({ name, category }, i) => {
    const id = name.replaceAll(' ', '_');
    return { i, name, things_id: id, tile: `img/tiles/${id}.svg`, cutout: `img/cutouts/${id}.svg`, category };
  });
const G = concepts.length;
const categorySize = Object.fromEntries(Object.entries(CATEGORIES).map(([name, members]) => [name, members.length]));

function write(path, content) {
  const full = join(ROOT, path);
  mkdirSync(dirname(full), { recursive: true });
  writeFileSync(full, typeof content === 'string' ? content : JSON.stringify(content));
}

const hue = (i) => Math.round((i * 137.508) % 360);
const shapeOf = (i) => `<g fill="hsl(${hue(i)} 48% 42%)">${SHAPES[i % SHAPES.length]}</g>`;
const svg = (body) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="256" height="256">${body}</svg>`;

const tileSvg = ({ i, name }) =>
  svg(`<rect width="100" height="100" fill="hsl(${hue(i)} 30% 86%)"/>${shapeOf(i)}` +
    `<text x="50" y="92" font-family="sans-serif" font-size="9" text-anchor="middle" fill="hsl(${hue(i)} 40% 22%)">${name}</text>`);

const cutoutSvg = ({ i }) => svg(shapeOf(i));

const impressionSvg = (members) =>
  svg('<filter id="b"><feGaussianBlur stdDeviation="3"/></filter><rect width="100" height="100" fill="#808080"/>' +
    `<g filter="url(#b)">${members.map((c) => `<g opacity="${round3(1 / members.length)}">${shapeOf(c)}</g>`).join('')}</g>`);

const u16 = (x) => Math.round(Math.min(Math.max(x, 0), 1) * U16);

function softmax(sims) {
  const avg = mean(sims);
  const sd = Math.sqrt(mean(sims.map((s) => (s - avg) ** 2)));
  const logits = sims.map((s) => (BETA * (s - avg)) / sd);
  const peak = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - peak));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / total);
}

function rankStep(q, truth) {
  const order = q.map((_, c) => c).sort((a, b) => q[b] - q[a] || a - b);
  const sorted = order.map((c) => q[c]);
  const excl = [];
  sorted.reduce((above, value) => (excl.push(above), above + value), 0);
  return { order, sorted, excl, trueExcl: excl[order.indexOf(truth)] };
}

const level = (j) => LAM[j];
const isFull = (j) => LAM[j] >= 1;

function sizeAt(step, j) {
  if (isFull(j)) return G;
  let lo = 0;
  let hi = G;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (step.excl[mid] < level(j)) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

const missAt = (step, j) => !isFull(j) && step.trueExcl >= level(j);

function stopIndex(steps, j) {
  if (isFull(j)) return K - 1;
  const t = steps.findIndex((step) => step.excl[M] >= level(j));
  return t === -1 ? K - 1 : t;
}

function confusersOf(i, random) {
  const { category } = concepts[i];
  const pool = concepts.filter((c) => c.i !== i && (category === null || c.category === category)).map((c) => c.i);
  return shuffled(pool, random).slice(0, 4);
}

function simulateImage(subject, i, random) {
  const strength = SUBJECT_GAIN[subject] * SIGNAL * Math.exp(0.35 * gaussian(random));
  const means = new Array(G).fill(0);
  means[i] = strength;
  const confused = random() < CONFUSED_SHARE;
  confusersOf(i, random).forEach((c, k) => {
    means[c] = strength * (k === 0 && confused ? 1.2 : 0.45 + 0.35 * random());
  });
  return Array.from({ length: R }, () => {
    const sums = new Array(G).fill(0);
    return Array.from({ length: K }, (_, t) => {
      sums.forEach((_, c) => (sums[c] += NOISE * gaussian(random)));
      return rankStep(softmax(means.map((mu, c) => mu + sums[c] / (t + 1))), i);
    });
  });
}

function imageTables(streams) {
  const stops = LAM.map((_, j) => streams.map((steps) => stopIndex(steps, j)));
  const fixed = Object.fromEntries(FIXED_K.map((k) => [k, {
    loss: LAM.map((_, j) => mean(streams.map((steps) => Number(missAt(steps[k - 1], j))))),
    size: LAM.map((_, j) => mean(streams.map((steps) => sizeAt(steps[k - 1], j)))),
  }]));
  return {
    loss_count: LAM.map((_, j) => streams.filter((steps, r) => missAt(steps[stops[j][r]], j)).length),
    stop_x10: LAM.map((_, j) => Math.round(mean(stops[j].map((t) => t + 1)) * 10)),
    size_x10: LAM.map((_, j) => Math.round(mean(streams.map((steps, r) => sizeAt(steps[stops[j][r]], j))) * 10)),
    a_true_u16: streams.map((steps) => steps.map((step) => u16(step.trueExcl))),
    top3_u16: streams.map((steps) => steps.map((step) => u16(step.excl[M]))),
    fixed,
  };
}

function mockIndex(rows, calIdx) {
  const first = LAM.findIndex((_, j) => mean(calIdx.map((i) => rows[i][j])) > MOCK_RISK_CAP);
  return Math.max((first === -1 ? LAM.length : first) - 1, 0);
}

const withArticle = (noun) => `${'aeiou'.includes(noun[0]) ? 'an' : 'a'} ${noun}`;
const categoryPhrase = (category) =>
  MASS_SUFFIXES.some((suffix) => category.endsWith(suffix)) ? category : withArticle(category);

function sharedCategory(members) {
  const category = concepts[members[0]].category;
  const shared = category !== null && members.every((c) => concepts[c].category === category);
  return shared && categorySize[category] <= MAX_CATEGORY_SIZE ? category : null;
}

function displayClaim(members) {
  if (members.length === G) return { kind: 'full', text: `Not sure yet: it could be any of the ${G}.`, category: null };
  if (members.length === 1) return { kind: 'single', text: `It's ${withArticle(concepts[members[0]].name)}.`, category: null };
  const category = sharedCategory(members);
  if (category) return { kind: 'category', text: `It's ${categoryPhrase(category)}.`, category };
  return { kind: 'list', text: `It's one of these ${members.length}.`, category: null };
}

function narrate(claim, members) {
  const names = members.map((c) => concepts[c].name);
  if (claim.kind === 'single') return `The picture showed ${withArticle(names[0])}.`;
  if (claim.kind === 'category') return `The picture showed ${categoryPhrase(claim.category)}.`;
  if (claim.kind === 'full') return `The picture could still be any of the ${G} objects.`;
  if (names.length <= 4) return `The picture showed ${names.slice(0, -1).map(withArticle).join(', ')} or ${withArticle(names.at(-1))}.`;
  return `The picture showed one of these ${names.length} objects.`;
}

function provenance(claim, members, narration, random) {
  const names = members.map((c) => concepts[c].name);
  const lead = concepts[members[0]];
  const confidence = round3(0.85 + 0.14 * random());
  const box = { x1: 60 + Math.round(random() * 80), y1: 70 + Math.round(random() * 80), x2: 620, y2: 600 };
  return {
    spark: {
      request: { model: 'muse-spark-1.3', instructions: SPARK_SYSTEM, input: `Claim: ${claim.text} Names: ${names.slice(0, 8).join(', ')}.`, stream: false, store: false },
      response: {
        id: `resp_mock${pad(Math.floor(random() * 1e8), 8)}`, object: 'response', model: 'muse-spark-1.3', status: 'completed',
        output: [{ type: 'message', role: 'assistant', content: [{ type: 'output_text', text: narration }] }],
        usage: { input_tokens: 96 + names.length * 3, output_tokens: 14 },
      },
    },
    sam: {
      request: {
        model: 'sam-3.1',
        input: [{ type: 'message', role: 'user', content: [
          { type: 'input_text', text: lead.name },
          { type: 'input_image', image_url: `data:image/webp;base64,(CC0 photo of ${lead.things_id}, 768 px, omitted)` },
        ] }],
        metadata: { include_confidence: 'true' }, stream: false, store: false,
      },
      response: {
        id: `resp_mock${pad(Math.floor(random() * 1e8), 8)}`, object: 'response', model: 'sam-3.1', status: 'completed',
        output: [{ type: 'message', role: 'assistant', content: [{ type: 'output_text',
          text: `<0f>0<|box;x1=${box.x1};y1=${box.y1};x2=${box.x2};y2=${box.y2};w=768;h=768;c=${confidence}|><|mask;x=0;y=0;c=${confidence};data=(run-length mask omitted)|>` }] }],
      },
    },
  };
}

const bump = (ms, centre, width, height) => height * Math.exp(-(((ms - centre) / width) ** 2));
const ERP = Array.from({ length: 50 }, (_, k) => {
  const ms = -200 + k * 20;
  return bump(ms, 100, 25, 1) - bump(ms, 170, 30, 1.6) + bump(ms, 250, 40, 1.1) + bump(ms, 420, 90, 0.5);
});

function eegTrace(random) {
  const sums = new Array(ERP.length).fill(0);
  const averages = Array.from({ length: K }, (_, t) => {
    const phase = random() * 2 * Math.PI;
    const white = ERP.map(() => gaussian(random));
    sums.forEach((_, k) => {
      const smooth = (white[Math.max(k - 1, 0)] + white[k] + white[Math.min(k + 1, ERP.length - 1)]) / 3;
      sums[k] += ERP[k] + 1.6 * Math.sin(phase + (2 * Math.PI * k) / 5) + 2.2 * smooth;
    });
    return sums.map((sum) => sum / (t + 1));
  });
  const scale = 120 / Math.max(...averages[0].map(Math.abs));
  return averages.map((row) => row.map((v) => Math.max(-127, Math.min(127, Math.round(v * scale)))));
}

function exampleFor(subject, i, streams, lamIdx, random) {
  const steps = streams[0];
  const sizes = steps.map((step) => sizeAt(step, lamIdx));
  const stoppedAt = stopIndex(steps, lamIdx) + 1;
  const final = steps[stoppedAt - 1];
  const members = final.order.slice(0, sizes[stoppedAt - 1]);
  const claim = displayClaim(members);
  const narration = narrate(claim, members);
  const impression = `img/impressions/sub-${pad(subject, 2)}/${pad(i, 3)}.svg`;
  write(impression, impressionSvg(members.slice(0, IMPRESSION_MEMBERS)));
  return {
    top: steps.map((step) => step.order.slice(0, TOP).map((c, k) => [c, u16(step.sorted[k])])),
    set_size: sizes,
    stopped_at: stoppedAt,
    eeg: eegTrace(random),
    claim,
    narration,
    provenance: provenance(claim, members, narration, random),
    pe_guess: concepts[final.order[0]].name,
    impression,
    truth_in_set: members.includes(i),
  };
}

const finalSize = (ex) => ex.set_size[ex.stopped_at - 1];

const CURATED_RULES = [
  ['clean stop', (i, ex) => ex.claim.kind === 'single' && ex.truth_in_set],
  ['unsure', (i, ex) => ex.stopped_at === K && finalSize(ex) > M],
  ['hedged, top pick wrong', (i, ex) => ex.truth_in_set && finalSize(ex) > 1 && ex.top[ex.stopped_at - 1][0][0] !== i],
  ['missed', (i, ex) => !ex.truth_in_set],
];

function pickCurated(subject, examples) {
  return CURATED_RULES.flatMap(([label, rule]) => {
    const hit = [...examples].find(([i, ex]) => rule(i, ex));
    return hit ? [{ subject, i: hit[0], label }] : [];
  });
}

function summarise(tables, cal, evalIdx, lamIdx, examples) {
  const atEval = (rows, j) => mean(evalIdx.map((i) => rows[i][j]));
  const losses = tables.map((t) => t.loss_count.map((count) => count / R));
  const fixed = FIXED_K.map((k) => {
    const loss = tables.map((t) => t.fixed[k].loss);
    const j = mockIndex(loss, cal);
    return { k, risk: atEval(loss, j), size: atEval(tables.map((t) => t.fixed[k].size), j) };
  });
  const meanStop = atEval(tables.map((t) => t.stop_x10), lamIdx) / 10;
  const meanSize = atEval(tables.map((t) => t.size_x10), lamIdx) / 10;
  const matched = fixed.find((f) => f.size <= meanSize) ?? fixed.at(-1);
  const kinds = [...examples.values()].map((ex) => ex.claim.kind);
  const share = (kind) => round3(kinds.filter((k) => k === kind).length / kinds.length);
  return {
    eval_risk: round3(atEval(losses, lamIdx)),
    conservativeness: round3(ALPHA - mean(losses.map((row) => row[lamIdx]))),
    mean_stop: round3(meanStop),
    mean_set_size: round3(meanSize),
    fixed_points: fixed.map((f) => [round3(f.size), f.k]),
    fixed_eval_risk: Object.fromEntries(fixed.map((f) => [String(f.k), round3(f.risk)])),
    h1_ratio: round3(meanStop / matched.k),
    h2a: { cp_upper: MOCK_CP_UPPER },
    outcome_mix: { single: share('single'), category: share('category'), list: share('list'), full: share('full') },
  };
}

function buildSubject(subject) {
  const random = mulberry32(1000 + subject);
  const streams = concepts.map(({ i }) => simulateImage(subject, i, random));
  const tables = streams.map(imageTables);
  const order = shuffled(concepts.map((c) => c.i), random);
  const cal = order.slice(0, 100).sort((a, b) => a - b);
  const evalIdx = order.slice(100).sort((a, b) => a - b);
  const lamIdx = mockIndex(tables.map((t) => t.loss_count.map((count) => count / R)), cal);
  const examples = new Map(evalIdx.map((i) => [i, exampleFor(subject, i, streams[i], lamIdx, random)]));
  examples.forEach((example, i) => write(`examples/sub-${pad(subject, 2)}/${pad(i, 3)}.json`, example));
  write(`tables/sub-${pad(subject, 2)}.json`, {
    loss_count: tables.map((t) => t.loss_count),
    stop_x10: tables.map((t) => t.stop_x10),
    size_x10: tables.map((t) => t.size_x10),
    naive: { a_true_u16: tables.map((t) => t.a_true_u16), top3_u16: tables.map((t) => t.top3_u16) },
  });
  return {
    pinned: { cal, eval: evalIdx, lam_idx: lamIdx },
    curated: pickCurated(subject, examples),
    summary: summarise(tables, cal, evalIdx, lamIdx, examples),
  };
}

rmSync(ROOT, { recursive: true, force: true });
concepts.forEach((concept) => {
  write(concept.tile, tileSvg(concept));
  write(concept.cutout, cutoutSvg(concept));
});
const built = SUBJECTS.map(buildSubject);
const signTestP = 0.5 ** SUBJECTS.length;
write('manifest.json', {
  mock: true,
  alpha: ALPHA,
  delta: DELTA,
  K,
  m: M,
  R,
  lam: LAM,
  subjects: SUBJECTS,
  demo_subjects: SUBJECTS,
  muse_image: true,
  pinned: Object.fromEntries(built.map((b, k) => [String(SUBJECTS[k]), b.pinned])),
  concepts,
  curated: built.flatMap((b) => b.curated).filter((c, k, all) => all.findIndex((d) => d.label === c.label) === k),
  results: {
    per_subject: Object.fromEntries(built.map((b, k) => [String(SUBJECTS[k]), b.summary])),
    pvalues: { H1: MOCK_H1_P, H2b: signTestP },
    holm: { H1: mean(built.map((b) => b.summary.h1_ratio)) <= 0.7, H2b: signTestP <= 0.05 },
    h2b_count: SUBJECTS.length,
  },
});
built.forEach((b, k) =>
  console.log(`sub-${SUBJECTS[k]}: lam_idx ${b.pinned.lam_idx}, stop ${b.summary.mean_stop}, size ${b.summary.mean_set_size}, ratio ${b.summary.h1_ratio}, mix ${JSON.stringify(b.summary.outcome_mix)}, curated ${b.curated.map((c) => `${c.label}:${c.i}`).join(' ')}`),
);
