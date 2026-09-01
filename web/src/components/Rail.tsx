import type { SolveEvent } from '../types'

type Props = {
  round: number
  filled: string
  icr: number
  calls: number
  tokens: number
  log: SolveEvent[]
}

export function Rail({ round, filled, icr, calls, tokens, log }: Props) {
  return (
    <aside className="rail">
      <h2>Solve</h2>
      <div className="stats">
        <div>
          Round <b>{round}</b>
        </div>
        <div>
          Slots <b>{filled}</b>
        </div>
        <div>
          ICR <b>{icr.toFixed(2)}</b>
        </div>
        <div>
          Calls <b>{calls}</b>
        </div>
        <div>
          Tokens <b>{tokens.toLocaleString()}</b>
        </div>
      </div>
      <ol className="log">
        {log.slice(-24).map((event, index) => (
          <li key={`${event.kind}-${event.round}-${index}`}>
            {event.message || event.kind}
          </li>
        ))}
      </ol>
    </aside>
  )
}
