@echo off
REM Set the base command parts
set SCRIPT_NAME=.\si_cluster_da_tpr_multidim.py
set ARG1=1
set ARG2=120

REM Run with the different third arguments
echo Running with parameter 100...
python %SCRIPT_NAME% %ARG1% %ARG2% 100

echo Running with parameter 150...
python %SCRIPT_NAME% %ARG1% %ARG2% 150

echo Running with parameter 200...
python %SCRIPT_NAME% %ARG1% %ARG2% 200

echo Running with parameter 250...
python %SCRIPT_NAME% %ARG1% %ARG2% 250

echo All runs complete.
pause