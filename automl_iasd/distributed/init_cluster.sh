yum install git
git clone https://github.com/alban12/AutoML-IASD.git 
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
cd /home/hadoop/AutoML-IASD/automl_iasd
poetry shell 
poetry build 


