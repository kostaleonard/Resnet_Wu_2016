dataset_args='{"dataset_fraction": 0.01}'
network_args='{"architecture": "lenet", "input_shape": [128, 128, 3], "num_classes": 1000}'
train_args='{"batch_size": 32, "epochs": 10, "augment_val": true, "early_stopping": false, "overfit_single_batch": false, "shuffle_on_epoch_end": true, "use_wandb": false, "save_model": false}'

all: train

train:
	@echo Training model.
	PYTHONPATH=$(PYTHONPATH):. python3 training/train_model.py --gpu 0 --dataset_args $(dataset_args) --network_args $(network_args) --train_args $(train_args)

pytest:
	@echo Running linting scripts.
	-pylint dataset
	-pylint models
	-pylint test
	-pylint training
	-pylint util
	@echo Running unit tests.
	-pytest test/*.py
