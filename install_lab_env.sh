echo "Creating virtual environment under ./env"
python3 -m venv ./env # Create a virtual environment (python3)
echo "Install requirements"
source env/bin/activate
which python
pip install -r requirements.txt --no-cache-dir
deactivate
echo "Finished und unset python path"
