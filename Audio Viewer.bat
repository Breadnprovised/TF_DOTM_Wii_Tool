@echo off
rem Audio Viewer - launches scripts\dotm_audio_gui.py
rem Drag a game folder onto this file, or double-click and pick the folder.
setlocal
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
%PY% "%~dp0scripts\dotm_audio_gui.py" %*
if errorlevel 1 (
    echo.
    echo Audio Viewer stopped with an error ^(see above^).
    echo Is Python 3 installed with: pip install numpy pillow ?
    pause
)
