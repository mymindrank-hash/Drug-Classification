.PHONY: install format train eval test all

install:
	python -m pip install -r requirements.txt

format:
	black --check train.py

train:
	python train.py

eval:
	cat results/metrics.txt

test:
	python verify_inference.py

all: install format train eval test
