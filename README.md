# AutoML-IASD
A distributed AutoML system for the IASD master's thesis module.

## Installation

You will need poetry to use the project, you can install it with pip if you don't have it : 

```bash
pip install poetry 
```

You can then install the dependency from the `pyproject.toml` file.

```bash
poetry install 
```

## Usage

To use the project, you will first need to get on a appropriate virtual environment. This can be done with the following command : 

```bash
poetry shell 
```

Then, there are two options :
- Use the program as a command line interface.
- Use the program as a library in your script.

### CLI

The CLI is runned from the `run_pipeline.py` script. 
It has 5 arguments : 
- dataset : Which is one of the provided datasets in the datasets folder.
- label_column_name : Which is the name of column to predict in that label
- task : Which is the type of task to perform ("Classification", "Regression")
- budget : Which is the budget allocate to the search of the best model.

Assuming that you are in the main folder and that you want to predict the Delay for the `airlines` dataset, you can then run it with : 

```bash
python AutoML-IASD/run_pipeline\ 
	--dataset airlines\ 
	--label_column_name Delay\
	--task classification\ 
	--budget 2
```

## Licence

