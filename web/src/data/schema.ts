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

export type Results = Record<string, unknown>;

export interface Manifest {
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
  provenance: { spark?: ApiExchange; sam?: ApiExchange } | null;
  pe_guess: string;
  composite: string[] | null;
  impression: string | null;
}
