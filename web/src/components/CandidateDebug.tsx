import { useMemo, useState } from 'react'
import type { CandidateBatch, PuzzleDetail } from '../types'

type Props = {
  puzzle: PuzzleDetail
  batches: CandidateBatch[]
  assignment: Record<string, string>
  showGold?: boolean
}

type Filter = 'all' | 'miss' | 'hit'

function slotLabel(id: string, puzzle: PuzzleDetail): string {
  const clue = [...puzzle.clues.across, ...puzzle.clues.down].find((item) => item.id === id)
  return clue ? `${clue.number}. ${clue.clue}` : id
}

export function CandidateDebug({ puzzle, batches, assignment, showGold = true }: Props) {
  const [filter, setFilter] = useState<Filter>('all')

  const rows = useMemo(() => {
    const out = []
    for (const batch of batches) {
      for (const slot of batch.slots) {
        const assigned = assignment[slot.id]
        if (filter === 'miss' && slot.hit !== false) continue
        if (filter === 'hit' && slot.hit !== true) continue
        out.push({ round: batch.round, assigned, ...slot })
      }
    }
    return out
  }, [batches, assignment, filter])

  const misses = batches.reduce(
    (n, batch) => n + batch.slots.filter((slot) => slot.hit === false).length,
    0,
  )
  const hits = batches.reduce(
    (n, batch) => n + batch.slots.filter((slot) => slot.hit === true).length,
    0,
  )

  return (
    <section className="debug">
      <div className="debug-head">
        <h2>Candidates</h2>
        <p>
          {showGold
            ? 'HIT means gold was in this batch. A miss here is a generation failure; a hit that still scores wrong is a search/repair failure.'
            : 'No answer key on this puzzle, so these are the model lists only.'}
        </p>
        {showGold ? (
          <div className="debug-filters" role="group" aria-label="Filter candidates">
            <button type="button" className={filter === 'all' ? 'on' : ''} onClick={() => setFilter('all')}>
              All
            </button>
            <button type="button" className={filter === 'miss' ? 'on' : ''} onClick={() => setFilter('miss')}>
              Miss {misses}
            </button>
            <button type="button" className={filter === 'hit' ? 'on' : ''} onClick={() => setFilter('hit')}>
              Hit {hits}
            </button>
          </div>
        ) : null}
      </div>
      {batches.length === 0 ? (
        <p className="debug-empty">Waiting for candidate batches from the model…</p>
      ) : (
      <table>
        <thead>
          <tr>
            <th>Rnd</th>
            <th>Slot</th>
            <th>Clue</th>
            <th>Pattern</th>
            {showGold ? <th>Gold</th> : null}
            {showGold ? <th></th> : null}
            <th>Assigned</th>
            <th>Candidates</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row.round}-${row.id}-${index}`} className={row.hit === false ? 'miss' : row.hit ? 'hit' : ''}>
              <td>{row.round}</td>
              <td className="mono">{row.id}</td>
              <td>{slotLabel(row.id, puzzle)}</td>
              <td className="mono">{row.pattern}</td>
              {showGold ? <td className="mono">{row.gold ?? '—'}</td> : null}
              {showGold ? (
                <td>{row.hit === true ? 'HIT' : row.hit === false ? 'MISS' : ''}</td>
              ) : null}
              <td className="mono">{row.assigned ?? '—'}</td>
              <td>
                {!row.candidates?.length
                  ? '—'
                  : row.candidates.map((item) => (
                      <span
                        key={item.answer}
                        className={
                          row.gold && item.answer.toUpperCase() === row.gold.toUpperCase()
                            ? 'offer gold'
                            : 'offer'
                        }
                      >
                        {item.answer}
                        <small>
                          {typeof item.confidence === 'number' ? item.confidence.toFixed(2) : ''}
                        </small>
                      </span>
                    ))}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      )}
    </section>
  )
}
