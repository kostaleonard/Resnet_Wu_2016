all: train

train:
	@echo "Not yet implemented."
	# TODO training

pytest:
	@echo "Running linting scripts."
	-pylint dataset
	-pylint models
	-pylint test
	-pylint training
	-pylint util
	@echo "Running unit tests."
	-pytest test/test_ilsvrc_dataset.py
	# TODO unit tests
