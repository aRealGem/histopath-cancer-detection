CONFIG ?= configs/baseline.yaml
.PHONY: setup data train eval predict smoke sweep kaggle-push lint clean

setup:      ## install python deps (skip TF on Colab/Kaggle)
	pip install -r requirements.txt

data:       ## download+extract competition data locally (needs kaggle.json)
	bash scripts/download_data.sh ./data

train:      ## two-phase transfer training
	python -m src.train --config $(CONFIG)

eval:       ## AUROC on held-out val + ROC png
	python -m src.evaluate --config $(CONFIG)

predict:    ## write artifacts/submission.csv
	python -m src.predict --config $(CONFIG)

smoke:      ## end-to-end sanity run on ~2k patches -> submission.csv
	python -m src.train    --config configs/smoke.yaml
	python -m src.evaluate --config configs/smoke.yaml
	python -m src.predict  --config configs/smoke.yaml

sweep:      ## grid sweep -> artifacts/sweep_results.csv
	python scripts/sweep.py --sweep configs/sweep.yaml

kaggle-push:  ## push runner notebook to Kaggle as a GPU kernel (needs kaggle.json)
	bash scripts/kaggle_push.sh

lint:       ## byte-compile all sources (fast sanity check)
	python -m py_compile src/*.py scripts/sweep.py

clean:
	rm -rf artifacts __pycache__ src/__pycache__ scripts/__pycache__
