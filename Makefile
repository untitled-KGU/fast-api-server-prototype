VENV = .venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip
UVICORN = $(VENV)/bin/uvicorn
PORT = 8000

.PHONY: setup install dev

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(MAKE) install

install:
	$(PIP) install -r requirements.txt

dev:
	$(UVICORN) main:app --reload --port $(PORT)