import type { Clue } from '../types'

type Props = {
  title: string
  clues: Clue[]
  hot: Set<string>
}

export function ClueList({ title, clues, hot }: Props) {
  return (
    <section className="clues">
      <h2>{title}</h2>
      {clues.map((clue) => (
        <div key={clue.id} className={hot.has(clue.id) ? 'clue hot' : 'clue'}>
          <span className="n">{clue.number}</span>
          <span>{clue.clue}</span>
          <span className="len">{clue.length}</span>
        </div>
      ))}
    </section>
  )
}
