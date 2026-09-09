sudo apt-get update
apt install python3.12-venv
pyenv versions
pyenv install 3.12
pyenv local 3.12
cd brest_cancer_detection/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
