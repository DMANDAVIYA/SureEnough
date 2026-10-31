export type ClaimKind = 'single' | 'category' | 'list' | 'full';

export interface Claim {
  kind: ClaimKind;
  text: string;
  category: string | null;
}

export interface Concept {
  i: number;
  name: string;
  things_id: string;
  tile: string;
  cutout: string;
  category: string | null;
}

export interface PinnedSplit {
  cal: number[];
  eval: number[];
  lam_idx: number;
}

export type OutcomeMix = Record<ClaimKind, number>;

export interface SubjectSummary {
  eval_risk: number;
  conservativeness: number;
  mean_stop: number;
  mean_set_size: number;
  fixed_points: [size: number, k: number][];
  fixed_eval_risk: Record<string, number>;
  h1_ratio: number;
  h2a: { cp_upper: number };
  outcome_mix: OutcomeMix;
}

export interface Results {
  per_subject: Record<string, SubjectSummary>;
  pvalues: { H1: number; H2b: number };
  holm: { H1: boolean; H2b: boolean };
  h2b_count: number;
  pe_vs_clip?: { pe: number; clip: number };
}

export const CURATED_LABELS = ['clean stop', 'unsure', 'hedged, top pick wrong', 'missed'] as const;

export type CuratedLabel = (typeof CURATED_LABELS)[number];

export interface Curated {
  subject: number;
  i: number;
  label: CuratedLabel;
}

export interface Manifest {
  mock?: boolean;
  alpha: number;
  delta: number;
  K: number;
  m: number;
  R: number;
  lam: number[];
  subjects: number[];
  demo_subjects: number[];
  muse_image: boolean;
  pinned: Record<string, PinnedSplit>;
  concepts: Concept[];
  curated: Curated[];
  results: Results;
}

export interface Table {
  loss_count: number[][];
  stop_x10: number[][];
  size_x10: number[][];
  naive: {
    a_true_u16: number[][][];
    top3_u16: number[][][];
  };
}

export interface ApiExchange {
  request: unknown;
  response: unknown;
}

export interface Example {
  top: [idx: number, p_u16: number][][];
  set_size: number[];
  stopped_at: number;
  eeg: number[][];
  claim: Claim;
  narration: string | null;
  provenance: { spark?: ApiExchange | null; sam?: ApiExchange | null } | null;
  pe_guess: string;
  impression: string | null;
  truth_in_set: boolean;
}

export interface FixtureSplit {
  cal: number[];
  eval: number[];
  lam_idx: number;
  eval_risk: number;
  naive_eval_risk: number;
}

export interface LttFixture {
  alpha: number;
  delta: number;
  table: Table;
  splits: FixtureSplit[];
}
