.PHONY: all stop evaluation fire-alert

all:
	docker compose up -d --build

stop:
	docker compose down

evaluation:
	docker compose run --rm evaluation python run_evaluation.py

fire-alert:
	@echo "Triggering ML alert by running evaluation with degraded data"
	docker compose run --rm evaluation python run_evaluation.py
