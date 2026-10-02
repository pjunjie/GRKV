PYTHON := .venv/bin/python
SOURCES := grkv kvpress experiments evaluation scripts tests

.PHONY: format style test
format:
	.venv/bin/isort $(SOURCES)
	.venv/bin/black $(SOURCES)

style:
	.venv/bin/flake8 $(SOURCES)
	.venv/bin/mypy $(SOURCES) --check-untyped-defs
	$(PYTHON) scripts/check_headers.py

test:
	$(PYTHON) -m pytest tests -q
