PY ?= ../.venv/bin/python

.PHONY: install test demo sample plots serve live

install:            ## parent library, then this project
	$(PY) -m pip install -e .. -e ".[dev]"

test:
	$(PY) -m pytest -q

demo:               ## offline end-to-end run on the synthetic sample
	$(PY) -m volrisk.cli demo

sample:             ## regenerate data/sample_SPY.csv (deterministic)
	$(PY) scripts/make_sample.py

plots:
	$(PY) scripts/plot_surface.py

serve:              ## API on the sample data, http://localhost:8000/docs
	VOLRISK_SOURCE=sample $(PY) -m uvicorn volrisk.api:app --reload

live:               ## fetch a real SPY chain from Yahoo and calibrate
	$(PY) -m volrisk.cli build --symbol SPY
