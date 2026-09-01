import type { CellState, PuzzleDetail } from '../types'

type Props = {
  puzzle: PuzzleDetail
  fills: Record<string, CellState>
  hot: Set<string>
  scorecard: boolean
  tone?: 'agent' | 'key'
}

function key(r: number, c: number): string {
  return `${r},${c}`
}

export function Board({ puzzle, fills, hot, scorecard, tone = 'agent' }: Props) {
  const blocks = new Set(puzzle.grid.blocks.map(([r, c]) => key(r, c)))
  const numbers = new Map(puzzle.grid.numbers.map(([r, c, n]) => [key(r, c), n]))
  const { height, width } = puzzle.grid

  return (
    <div
      className="board"
      style={{ gridTemplateColumns: `repeat(${width}, 1fr)` }}
      role="grid"
      aria-label={`${height} by ${width} crossword`}
    >
      {Array.from({ length: height }, (_, r) =>
        Array.from({ length: width }, (_, c) => {
          const id = key(r, c)
          if (blocks.has(id)) {
            return <div key={id} className="square block" role="gridcell" aria-label="block" />
          }
          const fill = fills[id]
          const classes = ['square']
          if (tone === 'key') classes.push('key')
          if (hot.has(id)) classes.push('hot')
          if (scorecard && tone === 'agent' && fill?.correct === true) classes.push('ok')
          if (scorecard && tone === 'agent' && fill?.correct === false) classes.push('bad')
          return (
            <div key={id} className={classes.join(' ')} role="gridcell">
              {numbers.has(id) ? <span className="num">{numbers.get(id)}</span> : null}
              <span className="glyph">{fill?.letter ?? ''}</span>
            </div>
          )
        }),
      )}
    </div>
  )
}
