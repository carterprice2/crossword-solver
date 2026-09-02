export type Suite = {
  id: string
  label: string
  count: number
  description: string
}

export type PuzzleSummary = {
  id: string
  suite: string
  title: string
  author: string
  height: number
  width: number
  size: string
  slots: number
  provenance: string
  has_gold: boolean
}

export type Clue = {
  id: string
  number: number
  clue: string
  length: number
  cells: [number, number][]
}

export type PuzzleDetail = {
  id: string
  title: string
  author: string
  provenance: string
  height: number
  width: number
  size: string
  slots: number
  has_gold: boolean
  grid: {
    height: number
    width: number
    blocks: [number, number][]
    numbers: [number, number, number][]
  }
  clues: {
    across: Clue[]
    down: Clue[]
  }
}

export type Defaults = {
  backend: string
  arm: string
  model: string
  models: string[]
  repair_model: string
  ensemble_model: string
  has_key: boolean
  arms: { id: string; label: string; description: string }[]
}

export type Scores = {
  wcr: number
  lcr: number
  icr: number
  exact: boolean
  cells_filled: number
  cells_correct: number
  cells_total: number
  slots_correct: number
  slots_total: number
}

export type CellMark = {
  r: number
  c: number
  letter: string
  correct?: boolean
}

export type CandidateOffer = {
  answer: string
  confidence: number
}

export type CandidateSlot = {
  id: string
  pattern: string
  candidates: CandidateOffer[]
  gold?: string
  hit?: boolean
}

export type CandidateBatch = {
  round: number
  slots: CandidateSlot[]
}

export type SolveEvent = {
  kind: string
  round: number
  message: string
  data: Record<string, unknown>
  scores?: Scores
  cells?: CellMark[]
  gold?: CellMark[]
  locked?: [number, number][]
  assignment?: Record<string, string>
  candidate_batches?: CandidateBatch[]
  solve?: { calls: number; seconds: number; prompt_tokens?: number; completion_tokens?: number }
}

export type IngestResponse = {
  status: 'ready' | 'needs_edit'
  draft_id?: string
  job_id?: string
  puzzle?: PuzzleDetail | null
  rows?: string[]
  height?: number
  width?: number
  across_slots?: number
  down_slots?: number
  across_clues?: number
  down_clues?: number
  unknown_numbers?: string[]
  message?: string
}

export type CellState = {
  letter: string
  correct: boolean | null
}
