all: train

train:
	@echo Training model.
	PYTHONPATH=$(PYTHONPATH):. python3 training/train_model.py

pytest:
	@echo Running linting scripts.
	-pylint dataset
	-pylint models
	-pylint test
	-pylint training
	-pylint util
	@echo Running unit tests.
	-pytest test/*.py
