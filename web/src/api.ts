import type { Defaults, IngestResponse, PuzzleDetail, PuzzleSummary, Suite } from './types'

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url)
  if (!response.ok) {
    const detail = await errorDetail(response)
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

async function errorDetail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string }
    if (body.detail) return body.detail
  } catch {
    /* not JSON */
  }
  return `${response.status} ${response.statusText}`
}

export function fetchSuites(): Promise<Suite[]> {
  return getJson('/api/suites')
}

export function fetchDefaults(): Promise<Defaults> {
  return getJson('/api/defaults')
}

export function fetchPuzzles(suite: string): Promise<PuzzleSummary[]> {
  return getJson(`/api/puzzles?suite=${encodeURIComponent(suite)}`)
}

export function fetchPuzzle(id: string): Promise<PuzzleDetail> {
  return getJson(`/api/puzzles/${encodeURIComponent(id)}`)
}

export async function startSolve(body: {
  puzzle_id: string
  arm: string
  model?: string
  ensemble_model?: string
}): Promise<{ job_id: string; backend: string; arm: string; model: string }> {
  const response = await fetch('/api/solves', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    throw new Error(await errorDetail(response))
  }
  return response.json() as Promise<{
    job_id: string
    backend: string
    arm: string
    model: string
  }>
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    throw new Error(await errorDetail(response))
  }
  return response.json() as Promise<T>
}

export function ingestPuzzle(body: {
  image?: string
  across?: string
  down?: string
  title?: string
  xd?: string
  arm?: string
  model?: string
  ensemble_model?: string
}): Promise<IngestResponse> {
  return postJson('/api/ingest', body)
}

export function fixIngestGrid(
  draftId: string,
  body: {
    rows: string[]
    across?: string
    down?: string
    arm?: string
    model?: string
    ensemble_model?: string
  },
): Promise<IngestResponse> {
  return postJson(`/api/ingest/${encodeURIComponent(draftId)}/grid`, body)
}
