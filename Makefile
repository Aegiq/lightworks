# Makefile
.DEFAULT_GOAL := run

format:
	autopep8 -i -r -a -a -a --max-line-length 79 lightworks/

check:
	pycodestyle lightworks/

mypy:
	mypy -p lightworks/