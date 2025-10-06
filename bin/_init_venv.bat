:: INITIATE PYTHON VIRTUAL ENVIRONMENT

:: Create virtual environment for this project
py -3.10 -m venv venv

:: Activate virtual environment
CALL venv\Scripts\activate.bat

:: Install requirements
pip install -r requirements.txt
