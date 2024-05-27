echo ":: Preparing Enviroment File"

cp .env.example .env

echo ":: Installing Python Virtual Environment to working directory"

$(python -m venv venv)

echo ":: Activating Python Virtual Environment"
.\venv\Scripts\activate

echo ":: Installing Python packages"
$(pip install -r requirements.txt)

echo ":: Installation complete - run 'run.ps1' to execute the script."