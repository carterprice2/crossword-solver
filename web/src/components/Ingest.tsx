import { useState } from 'react'
import { fixIngestGrid, ingestPuzzle } from '../api'
import type { IngestResponse, PuzzleDetail } from '../types'

type Props = {
  busy: boolean
  arm: string
  model: string
  ensembleModel: string
  debug: boolean
  onError: (message: string | null) => void
  onReady: (puzzle: PuzzleDetail, jobId: string) => void
}

function toggleCell(rows: string[], r: number, c: number): string[] {
  return rows.map((row, i) => {
    if (i !== r) return row
    const cells = Array.from(row)
    const ch = cells[c]
    cells[c] = ch === '#' ? '.' : '#'
    return cells.join('')
  })
}

export function Ingest({ busy, arm, model, ensembleModel, debug, onError, onReady }: Props) {
  const [across, setAcross] = useState('')
  const [down, setDown] = useState('')
  const [xd, setXd] = useState('')
  const [image, setImage] = useState<string | null>(null)
  const [fileName, setFileName] = useState('')
  const [draft, setDraft] = useState<IngestResponse | null>(null)
  const [sending, setSending] = useState(false)

  const blocked = busy || sending

  async function onFile(file: File | undefined) {
    if (!file) return
    if (!/^image\/(png|jpeg)$/.test(file.type)) {
      onError('Screenshot must be a PNG or JPEG.')
      return
    }
    if (file.size > 4 * 1024 * 1024) {
      onError('Screenshot is larger than 4 MB.')
      return
    }
    const data = await readDataUrl(file)
    setImage(data)
    setFileName(file.name)
    onError(null)
  }

  async function submit() {
    if (blocked) return
    setSending(true)
    onError(null)
    try {
      const extra = arm === 'a4' && ensembleModel ? { ensemble_model: ensembleModel } : {}
      const body = xd.trim()
        ? { xd: xd.trim(), arm, model, debug, ...extra }
        : { image: image || undefined, across, down, arm, model, debug, ...extra }
      const result = await ingestPuzzle(body)
      handle(result)
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err))
    } finally {
      setSending(false)
    }
  }

  async function submitGrid() {
    if (!draft?.draft_id || !draft.rows || blocked) return
    setSending(true)
    onError(null)
    try {
      const extra = arm === 'a4' && ensembleModel ? { ensemble_model: ensembleModel } : {}
      const result = await fixIngestGrid(draft.draft_id, {
        rows: draft.rows,
        arm,
        model,
        debug,
        ...extra,
      })
      handle(result)
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err))
    } finally {
      setSending(false)
    }
  }

  function handle(result: IngestResponse) {
    if (result.status === 'ready' && result.puzzle && result.job_id) {
      setDraft(null)
      onReady(result.puzzle, result.job_id)
      return
    }
    setDraft(result)
  }

  const editing = draft?.status === 'needs_edit' && draft.rows?.length

  return (
    <section className="ingest">
      <div className="ingest-copy">
        <h2>Bring a crossword</h2>
        <p>
          Photo of the grid, then the Across and Down clues. If the black squares look wrong, toggle
          them before we start.
        </p>
      </div>
      <div className="ingest-form">
        <label className="drop">
          <input
            type="file"
            accept="image/png,image/jpeg"
            disabled={blocked}
            onChange={(e) => onFile(e.target.files?.[0])}
          />
          <span>{fileName || 'Drop a grid screenshot'}</span>
        </label>
        <label className="field grow">
          <span>Across</span>
          <textarea
            value={across}
            disabled={blocked}
            placeholder={'1. First across clue\n14. Next'}
            onChange={(e) => setAcross(e.target.value)}
          />
        </label>
        <label className="field grow">
          <span>Down</span>
          <textarea
            value={down}
            disabled={blocked}
            placeholder={'1. First down clue\n2. Next'}
            onChange={(e) => setDown(e.target.value)}
          />
        </label>
        <label className="field grow">
          <span>.xd paste</span>
          <textarea
            value={xd}
            disabled={blocked}
            placeholder="Optional. Skips the photo."
            onChange={(e) => setXd(e.target.value)}
          />
        </label>
        <button className="start" type="button" disabled={blocked || (!xd.trim() && !image)} onClick={submit}>
          {sending ? 'Reading grid' : 'Solve this'}
        </button>
      </div>
      {editing ? (
        <div className="ingest-edit">
          <p>
            {draft.message || 'Slot counts do not match the clues.'} Across {draft.across_clues}/
            {draft.across_slots}, down {draft.down_clues}/{draft.down_slots}. Click a square to
            toggle a block.
          </p>
          <div
            className="board edit-board"
            style={{ gridTemplateColumns: `repeat(${draft.width}, 1fr)` }}
            role="grid"
            aria-label="Edit block mask"
          >
            {draft.rows!.map((row, r) =>
              Array.from(row).map((ch, c) => (
                <button
                  key={`${r}-${c}`}
                  type="button"
                  className={ch === '#' ? 'square block' : 'square'}
                  disabled={blocked}
                  onClick={() =>
                    setDraft((prev) =>
                      prev?.rows ? { ...prev, rows: toggleCell(prev.rows!, r, c) } : prev,
                    )
                  }
                >
                  <span className="glyph">{ch === '#' || ch === '.' ? '' : ch}</span>
                </button>
              )),
            )}
          </div>
          <button className="start" type="button" disabled={blocked} onClick={submitGrid}>
            Use this grid
          </button>
        </div>
      ) : null}
    </section>
  )
}

function readDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(new Error('could not read the screenshot'))
    reader.readAsDataURL(file)
  })
}
