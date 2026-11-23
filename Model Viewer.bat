@echo off
rem Model Viewer - launches scripts\dotm_model_viewer.py
rem Drag a game folder onto this file, or double-click and pick the folder.
setlocal
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
%PY% "%~dp0scripts\dotm_model_viewer.py" %*
if errorlevel 1 (
    echo.
    echo Model Viewer stopped with an error ^(see above^).
    echo Is Python 3 installed with: pip install numpy pillow "pyglet>=2.1" ?
    pause
)
