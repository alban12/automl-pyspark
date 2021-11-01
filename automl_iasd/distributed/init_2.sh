sudo yum install git
git clone https://github.com/alban12/AutoML-IASD.git 
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
source $HOME/.poetry/env
cd /home/hadoop/AutoML-IASD
poetry shell 
poetry build 


