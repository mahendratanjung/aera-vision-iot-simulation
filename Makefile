# AERA Vision. Run from the repo root.
#   make data     download CrowdHuman val + model weights, with checksums (about 3.4 GB)
#   make eval     rerun the whole evaluation: HEAVY (about 90 min of full CPU and GPU on an M4 Pro;
#                 it overloaded and restarted the laptop on 13 Nov 2026). Prefer a free Colab GPU.
#   make score    score the saved model outputs and rewrite the report: light, no model runs
#   make figures  report figures for the chosen model into figures/ (light)
#   make demo     rebuild the app's camera photos from the test split (after make eval)
#   make app      run the app on http://localhost:3100
#   make live     local live counter for the app's Live camera panel (port 8765)
#   make check    lint, tests and typecheck for both halves

M = cd model && uv run
.PHONY: data eval infer bench score report figures demo app live check

data:
	cd model && bash scripts/get_data.sh

eval: infer bench score report

infer:
	$(M) python -m aera_vision.infer

bench:
	$(M) python -m aera_vision.bench

score:
	$(M) python -m aera_vision.evaluate --available-only
	$(M) python -m aera_vision.report

report:
	$(M) python -m aera_vision.report

figures:
	$(M) python -m aera_vision.figures

demo:
	$(M) python -m aera_vision.demo

app:
	cd app && npm run dev

live:
	$(M) python -m aera_vision.serve

check:
	cd model && uv run ruff check . && uv run pytest -q
	cd app && npm run check
