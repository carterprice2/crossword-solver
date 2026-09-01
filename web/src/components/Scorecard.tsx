import type { Scores } from '../types'

type Props = {
  scores: Scores
  seconds: number
  tokens: number
}

export function Scorecard({ scores, seconds, tokens }: Props) {
  return (
    <div className="scorecard">
      <div className="verdict">{scores.exact ? 'Solved' : 'Partial'}</div>
      <div className="metric" title="Word Coverage Rate: correct slots / total slots">
        <span>WCR</span>
        <b>{scores.wcr.toFixed(3)}</b>
      </div>
      <div className="metric" title="Letter Coverage Rate: correct letters / fillable cells">
        <span>LCR</span>
        <b>{scores.lcr.toFixed(3)}</b>
      </div>
      <div className="metric" title="Intersection Consistency Rate: crossings where across and down assignments agree. Needs no answer key.">
        <span>ICR</span>
        <b>{scores.icr.toFixed(3)}</b>
      </div>
      <div className="metric">
        <span>Letters</span>
        <b>
          {scores.cells_correct}/{scores.cells_total}
        </b>
      </div>
      <div className="metric">
        <span>Time</span>
        <b>{seconds.toFixed(1)}s</b>
      </div>
      <div className="metric">
        <span>Tokens</span>
        <b>{tokens.toLocaleString()}</b>
      </div>
      <dl className="glossary">
        <div>
          <dt>WCR</dt>
          <dd>Word Coverage Rate. Correct slots divided by total slots. Primary score.</dd>
        </div>
        <div>
          <dt>LCR</dt>
          <dd>Letter Coverage Rate. Correct letters divided by fillable cells.</dd>
        </div>
        <div>
          <dt>ICR</dt>
          <dd>
            Intersection Consistency Rate. Fraction of crossings where the across and down
            assignments agree. Needs no gold; a filled cell grid always scores 1, so this is
            scored on slot strings.
          </dd>
        </div>
      </dl>
    </div>
  )
}
