import { useEffect, useMemo, useRef, useState } from 'react'
import { fetchDefaults, fetchPuzzle, fetchPuzzles, fetchSuites, startSolve } from './api'
import { Board } from './components/Board'
import { CandidateDebug } from './components/CandidateDebug'
import { ClueList } from './components/ClueList'
import { Picker } from './components/Picker'
import { Rail } from './components/Rail'
import { Scorecard } from './components/Scorecard'
import type {
  CandidateBatch,
  CandidateSlot,
  CellState,
  Defaults,
  PuzzleDetail,
  PuzzleSummary,
  Scores,
  SolveEvent,
  Suite,
} from './types'

function cellKey(r: number, c: number): string {
  return `${r},${c}`
}

function absorbGrid(rows: string[], fills: Record<string, CellState>): Record<string, CellState> {
  const next = { ...fills }
  rows.forEach((row, r) => {
    Array.from(row).forEach((ch, c) => {
      if (ch === '#' || ch === '.' || ch === ' ') return
      const id = cellKey(r, c)
      next[id] = { letter: ch, correct: next[id]?.correct ?? null }
    })
  })
  return next
}

function marksToFills(
  cells: { r: number; c: number; letter: string; correct?: boolean }[],
  correctDefault: boolean | null = null,
): Record<string, CellState> {
  const next: Record<string, CellState> = {}
  for (const cell of cells) {
    next[cellKey(cell.r, cell.c)] = {
      letter: cell.letter,
      correct: cell.correct ?? correctDefault,
    }
  }
  return next
}

export function App() {
  const [suites, setSuites] = useState<Suite[]>([])
  const [defaults, setDefaults] = useState<Defaults | null>(null)
  const [suite, setSuite] = useState('mini')
  const [puzzles, setPuzzles] = useState<PuzzleSummary[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [puzzle, setPuzzle] = useState<PuzzleDetail | null>(null)
  const [backend, setBackend] = useState('oracle')
  const [arm, setArm] = useState('a3')
  const [model, setModel] = useState('')
  const [debug, setDebug] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [fills, setFills] = useState<Record<string, CellState>>({})
  const [goldFills, setGoldFills] = useState<Record<string, CellState> | null>(null)
  const [hotSlots, setHotSlots] = useState<string[]>([])
  const [log, setLog] = useState<SolveEvent[]>([])
  const [round, setRound] = useState(0)
  const [filled, setFilled] = useState('0')
  const [icr, setIcr] = useState(1)
  const [calls, setCalls] = useState(0)
  const [tokens, setTokens] = useState(0)
  const [scores, setScores] = useState<Scores | null>(null)
  const [seconds, setSeconds] = useState(0)
  const [scorecard, setScorecard] = useState(false)
  const [candidateBatches, setCandidateBatches] = useState<CandidateBatch[]>([])
  const [assignment, setAssignment] = useState<Record<string, string>>({})
  const [showDebug, setShowDebug] = useState(false)
  const solveGen = useRef(0)

  useEffect(() => {
    Promise.all([fetchSuites(), fetchDefaults()])
      .then(([suiteList, defs]) => {
        setSuites(suiteList)
        setDefaults(defs)
        setBackend(defs.backend)
        setArm(defs.arm)
        if (defs.model) setModel(defs.model)
      })
      .catch((err: Error) => setError(err.message))
  }, [])

  useEffect(() => {
    fetchPuzzles(suite)
      .then((list) => {
        setPuzzles(list)
        const first = list.find((p) => p.height === 7) ?? list[0]
        if (first) setSelected(first.id)
      })
      .catch((err: Error) => setError(err.message))
  }, [suite])

  useEffect(() => {
    if (!selected) return
    const gen = solveGen.current
    let cancelled = false
    setFills({})
    setGoldFills(null)
    setLog([])
    setScores(null)
    setScorecard(false)
    setHotSlots([])
    setRound(0)
    setCandidateBatches([])
    setAssignment({})
    setShowDebug(false)
    fetchPuzzle(selected)
      .then((detail) => {
        if (cancelled) return
        setPuzzle(detail)
        if (gen !== solveGen.current) return
        setFilled(`0/${detail.slots}`)
        setIcr(1)
        setCalls(0)
        setTokens(0)
        setError(null)
      })
      .catch((err: Error) => {
        if (!cancelled && gen === solveGen.current) setError(err.message)
      })
    return () => {
      cancelled = true
    }
  }, [selected])

  const hotCells = useMemo(() => {
    const cells = new Set<string>()
    if (!puzzle) return cells
    for (const clue of [...puzzle.clues.across, ...puzzle.clues.down]) {
      if (!hotSlots.includes(clue.id)) continue
      for (const [r, c] of clue.cells) cells.add(cellKey(r, c))
    }
    return cells
  }, [puzzle, hotSlots])

  const nytLive = suite === 'nyt' && backend === 'nebius'
  const currentSuite = suites.find((item) => item.id === suite)

  async function onSolve() {
    if (!selected || busy) return
    solveGen.current += 1
    setBusy(true)
    setError(null)
    setScores(null)
    setScorecard(false)
    setFills({})
    setGoldFills(null)
    setLog([])
    setCandidateBatches([])
    setAssignment({})
    setShowDebug(debug)
    try {
      const job = await startSolve({
        puzzle_id: selected,
        backend,
        arm,
        debug,
        model: backend === 'nebius' ? model : undefined,
      })
      const source = new EventSource(`/api/solves/${job.job_id}/events`)
      let done = false
      source.onmessage = (message) => {
        let event: SolveEvent
        try {
          event = JSON.parse(message.data) as SolveEvent
        } catch {
          return
        }
        setLog((prev) => [...prev, event])
        setRound(event.round)
        if (event.kind === 'search') {
          const grid = event.data.grid
          if (Array.isArray(grid)) {
            setFills((prev) => absorbGrid(grid as string[], prev))
          }
          if (typeof event.data.filled === 'number' && puzzle) {
            setFilled(`${event.data.filled}/${puzzle.slots}`)
          }
          if (typeof event.data.icr === 'number') setIcr(event.data.icr)
        }
        if (event.kind === 'batch_done') {
          setCalls((n) => n + 1)
          const tok = event.data.tokens
          if (typeof tok === 'number') setTokens((n) => n + tok)
        }
        if (event.kind === 'candidates') {
          const slots = event.data?.slots
          if (Array.isArray(slots)) {
            setCandidateBatches((prev) => [
              ...prev,
              { round: event.round, slots: slots as CandidateSlot[] },
            ])
          }
        }
        if (event.kind === 'repair') {
          const slots = event.data.slots
          if (Array.isArray(slots)) setHotSlots(slots as string[])
        }
        if (event.kind === 'finished') {
          done = true
          source.close()
          setBusy(false)
          setScorecard(true)
          if (event.scores) setScores(event.scores)
          if (event.solve?.seconds) setSeconds(event.solve.seconds)
          if (event.solve?.calls) setCalls(event.solve.calls)
          const prompt = event.solve?.prompt_tokens ?? 0
          const completion = event.solve?.completion_tokens ?? 0
          if (prompt || completion) setTokens(prompt + completion)
          if (event.assignment) setAssignment(event.assignment)
          if (event.cells) setFills(marksToFills(event.cells))
          if (event.gold) setGoldFills(marksToFills(event.gold, true))
          if (event.candidate_batches?.length) setCandidateBatches(event.candidate_batches)
        }
        if (event.kind === 'error') {
          done = true
          source.close()
          setBusy(false)
          setError(event.message || 'Solve failed')
        }
      }
      source.onerror = () => {
        if (done) {
          source.close()
          return
        }
        if (source.readyState === EventSource.CONNECTING) return
        source.close()
        setBusy(false)
        setError((prev) => prev ?? 'Lost the solve stream')
      }
    } catch (err) {
      setBusy(false)
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <div className="shell">
      <header className="masthead">
        <div className="brand">
          <span className="brand-kicker">Drafting table</span>
          <h1>Crossword Agent</h1>
        </div>
        <div className="controls">
          <label className="field">
            <span>Backend</span>
            <select value={backend} onChange={(e) => setBackend(e.target.value)} disabled={busy}>
              <option value="oracle">Oracle (offline)</option>
              <option value="nebius" disabled={defaults ? !defaults.has_key : false}>
                Nebius (live)
              </option>
            </select>
          </label>
          <label className="field">
            <span>Arm</span>
            <select value={arm} onChange={(e) => setArm(e.target.value)} disabled={busy}>
              {(defaults?.arms ?? [{ id: 'a3', label: 'full agent' }]).map((item) => (
                <option key={item.id} value={item.id}>
                  {item.id} · {item.label}
                </option>
              ))}
            </select>
          </label>
          {backend === 'nebius' ? (
            <label className="field wide">
              <span>Model</span>
              <select
                value={model || defaults?.model || ''}
                title={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={busy}
              >
                {(defaults?.models ?? []).map((id) => (
                  <option key={id} value={id} title={id}>
                    {id.includes('/') ? id.slice(id.lastIndexOf('/') + 1) : id}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          <label className="toggle">
            <input
              type="checkbox"
              checked={debug}
              disabled={busy}
              onChange={(e) => setDebug(e.target.checked)}
            />
            Debug candidates
          </label>
          <button className="start" type="button" onClick={onSolve} disabled={!selected || busy}>
            {busy ? 'Solving' : 'Solve'}
          </button>
        </div>
      </header>

      {nytLive ? (
        <p className="warn">
          Live 15×15 solves spend tokens and take a while. Oracle is free and still scores against
          gold.
        </p>
      ) : null}
      {currentSuite?.warning && suite === 'nyt' && !nytLive ? (
        <p className="warn">{currentSuite.warning}</p>
      ) : null}

      <Picker
        suites={suites}
        suite={suite}
        puzzles={puzzles}
        selected={selected}
        onSuite={setSuite}
        onSelect={setSelected}
      />

      {error ? <p className="error">{error}</p> : null}

      {puzzle ? (
        <>
          {showDebug ? (
            <CandidateDebug puzzle={puzzle} batches={candidateBatches} assignment={assignment} />
          ) : null}
          <div className="stage">
            <div className={goldFills ? 'boards compare' : 'boards'}>
              <div className="board-wrap">
                {goldFills ? <h2 className="board-label">Agent</h2> : null}
                <Board puzzle={puzzle} fills={fills} hot={hotCells} scorecard={scorecard} />
              </div>
              {goldFills ? (
                <div className="board-wrap">
                  <h2 className="board-label">Answer key</h2>
                  <Board
                    puzzle={puzzle}
                    fills={goldFills}
                    hot={new Set()}
                    scorecard={false}
                    tone="key"
                  />
                </div>
              ) : null}
            </div>
            <ClueList title="Across" clues={puzzle.clues.across} hot={new Set(hotSlots)} />
            <ClueList title="Down" clues={puzzle.clues.down} hot={new Set(hotSlots)} />
            <Rail round={round} filled={filled} icr={icr} calls={calls} tokens={tokens} log={log} />
          </div>
          {scores ? <Scorecard scores={scores} seconds={seconds} tokens={tokens} /> : null}
        </>
      ) : (
        <p className="empty">Pick a puzzle to load the grid.</p>
      )}
    </div>
  )
}
