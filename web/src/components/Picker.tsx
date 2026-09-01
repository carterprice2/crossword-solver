import type { PuzzleSummary, Suite } from '../types'

type Props = {
  suites: Suite[]
  suite: string
  puzzles: PuzzleSummary[]
  selected: string | null
  onSuite: (id: string) => void
  onSelect: (id: string) => void
}

export function Picker({ suites, suite, puzzles, selected, onSuite, onSelect }: Props) {
  return (
    <div className="picker">
      <div className="tabs" role="tablist" aria-label="Puzzle suite">
        {suites.map((item) => (
          <button
            key={item.id}
            className={item.id === suite ? 'tab on' : 'tab'}
            type="button"
            role="tab"
            aria-selected={item.id === suite}
            onClick={() => onSuite(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className="chips">
        {puzzles.map((puzzle) => (
          <button
            key={puzzle.id}
            className={puzzle.id === selected ? 'chip on' : 'chip'}
            type="button"
            onClick={() => onSelect(puzzle.id)}
          >
            <b>{puzzle.size}</b>
            <small>
              {puzzle.slots} clues · {puzzle.id}
            </small>
          </button>
        ))}
      </div>
    </div>
  )
}
