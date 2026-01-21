VENV = .venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip
UVICORN = $(VENV)/bin/uvicorn
PORT = 8000
CACHE = __pycache__
MODEL_PATH = models

.PHONY: setup install dev clean

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(MAKE) install

install:
	$(PIP) install -r requirements.txt

dev:
	$(UVICORN) main:app --reload --port $(PORT)

clean:
	rm -rf $(VENV)
	rm -rf $(MODEL_PATH)
	find . -name $(CACHE) -type d -exec rm -rf {} +