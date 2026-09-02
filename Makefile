# Everything here runs with the Python standard library alone.
# `make test` and `make demo` need neither network nor an API key.

PY ?= python3
SUITE ?= mini
ARMS ?= a0,a1,a2,a3
DEMO ?= corpus/mini/mini-11-04-0.xd
MODEL ?= Qwen/Qwen3-30B-A3B-Instruct-2507
REPAIR_MODEL ?= Qwen/Qwen3-235B-A22B-Instruct-2507

.PHONY: help test demo solve eval report sweep sweep-pattern corpus bank nyt models clean \
	verify-offline verify-live-ping verify-live-parse verify-live-smoke verify-live-pair verify-live-ablation \
	serve serve-dev web-build screen-arms screen-models final-grid

help:
	@echo "make test     run the test suite          (no network, no install)"
	@echo "make demo     watch a solve, offline      (no network, no install)"
	@echo "make solve    solve one puzzle on Nebius  (needs NEBIUS_API_KEY)"
	@echo "make eval     run the ablation matrix     (needs NEBIUS_API_KEY)"
	@echo "make screen-arms / screen-models / final-grid   staged live tournament"
	@echo "make sweep    regenerate the offline oracle sweep"
	@echo "make report   rebuild summary.md from a results dir (DIR=...)"
	@echo "make corpus   regenerate the committed puzzles"
	@echo "make bank     rebuild the word/clue bank from its public-domain sources"
	@echo "make nyt      write the local NYT Friday fixture (corpus/nyt/)"
	@echo "make models   check NEBIUS_API_KEY and list reachable models"
	@echo "make serve    open the web UI               (needs pip install -e '.[web]')"
	@echo "make serve-dev  API only; run Vite separately"
	@echo "make verify-offline        tests (no network)"
	@echo "make verify-live-ping      key + reachable models"
	@echo "make verify-live-parse     one 2-clue parse smoke per catalog model"
	@echo "make verify-live-smoke     one 7x7 live solve, recorded (cheap model)"
	@echo "make verify-live-pair      a2 vs a3 on one 7x7, one seed"
	@echo "make verify-live-ablation  a2 vs a3 on four 7x7s, repair model"

test:
	$(PY) -m unittest discover -s tests -t . -v

# The offline demo: synthetic candidates with 40% of the correct answers
# missing, so the repair rounds visibly do the work.
demo:
	FORCE_COLOR=1 $(PY) -m crossword solve $(DEMO) \
		--backend oracle --oracle-recall 0.6 --oracle-top1-error 0.5 --live

solve:
	$(PY) -m crossword solve $(DEMO) --live \
		--model $(MODEL) --repair-model $(REPAIR_MODEL)

eval:
	$(PY) -m crossword eval --suite $(SUITE) --arms $(ARMS) \
		--model $(MODEL) --repair-model $(REPAIR_MODEL)

# The same matrix against synthetic candidates: no key, no spend.
eval-offline:
	$(PY) -m crossword eval --suite $(SUITE) --arms $(ARMS) --backend oracle

# Pause-gated live tournament. After each stage, edit winners.json then run the next.
screen-arms:
	$(PY) -m crossword eval --recipe screen-arms \
		--model $(MODEL) --repair-model $(REPAIR_MODEL)

screen-models:
	@test -n "$(FROM)" || (echo "usage: make screen-models FROM=results/run-..." && exit 1)
	$(PY) -m crossword eval --recipe screen-models --from $(FROM) \
		--model $(MODEL) --repair-model $(REPAIR_MODEL)

final-grid:
	@test -n "$(FROM)" || (echo "usage: make final-grid FROM=results/run-..." && exit 1)
	$(PY) -m crossword eval --recipe final-grid --from $(FROM) \
		--model $(MODEL) --repair-model $(REPAIR_MODEL)

sweep:
	$(PY) scripts/oracle_sweep.py --independent --out results/synthetic-sweep.json

sweep-pattern:
	$(PY) scripts/oracle_sweep.py --out results/pattern-aware-sweep.json

report:
	@test -n "$(DIR)" || (echo "usage: make report DIR=results/run-..." && exit 1)
	$(PY) -m crossword report $(DIR)

corpus:
	$(PY) scripts/make_corpus.py --out corpus

bank:
	$(PY) scripts/build_bank.py --out corpus/bank/words.tsv

nyt:
	$(PY) scripts/write_nyt_2021_05_28.py

models:
	$(PY) -m crossword models ping

serve:
	$(PY) -m crossword serve --build --host 127.0.0.1 --port 8000

serve-dev:
	$(PY) -m crossword serve --host 127.0.0.1 --port 8000

web-build:
	cd web && npm run build

# Staged verification. Offline is free. Live stages spend tokens; run in order.
verify-offline:
	$(PY) -m unittest discover -s tests -t . -q

verify-live-ping:
	$(PY) -m crossword models ping

verify-live-parse:
	$(PY) -m crossword models smoke

verify-live-smoke:
	mkdir -p results
	$(PY) -m crossword solve corpus/mini/mini-07-00-0.xd \
		--arm a3 --model $(MODEL) --repair-model $(MODEL) \
		--record results/live-smoke.jsonl

verify-live-pair:
	$(PY) -m crossword eval --suite mini --limit 1 --arms a2,a3 --seeds 1 \
		--model $(MODEL) --repair-model $(MODEL) \
		--run-id live-pair-7x7 --out results

verify-live-ablation:
	$(PY) -m crossword eval --suite mini --limit 4 --arms a2,a3 --seeds 1 \
		--model $(MODEL) --repair-model $(REPAIR_MODEL) \
		--run-id live-ablation-7x7 --out results

clean:
	rm -rf results/run-* __pycache__ */__pycache__ */*/__pycache__
