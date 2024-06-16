@ECHO OFF

pushd %~dp0

REM 

if "%1%" == "format" (
	autopep8 -i -r -a -a --max-line-length 79 lightworks/
)

if "%1%" == "check" (
	pycodestyle lightworks/
)

if "%1%" == "mypy" (
	mypy -p lightworks
)

:end
popd
