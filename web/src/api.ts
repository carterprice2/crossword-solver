import type { Defaults, PuzzleDetail, PuzzleSummary, Suite } from './types'

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
  backend: string
  arm: string
  model?: string
  debug?: boolean
}): Promise<{ job_id: string; backend: string; arm: string; model: string; debug: boolean }> {
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
    debug: boolean
  }>
}
