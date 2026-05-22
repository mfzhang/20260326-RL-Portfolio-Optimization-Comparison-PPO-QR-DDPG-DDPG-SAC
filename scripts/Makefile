# Makefile for RL Portfolio Optimization
# Run from the project root: make -f scripts/Makefile <target>

COMPOSE := docker-compose

.PHONY: help install install-dev test lint format \
        docker-build docker-up docker-down docker-logs docker-restart \
        train evaluate figures analyze-costs analyze-reward analyze-regime \
        api api-prod clean clean-data db-init db-backup db-restore \
        notebook shell setup full-pipeline

help:
	@echo "RL Portfolio Optimization — Available Commands"
	@echo ""
	@echo "  Installation"
	@echo "    make -f scripts/Makefile install        Install core dependencies"
	@echo "    make -f scripts/Makefile install-dev    Install dev dependencies"
	@echo ""
	@echo "  Quality"
	@echo "    make -f scripts/Makefile test           Run test suite"
	@echo "    make -f scripts/Makefile lint           Run linters"
	@echo "    make -f scripts/Makefile format         Auto-format code"
	@echo ""
	@echo "  Docker  (run from project root)"
	@echo "    make -f scripts/Makefile docker-build   Build images"
	@echo "    make -f scripts/Makefile docker-up      Start all services"
	@echo "    make -f scripts/Makefile docker-down    Stop all services"
	@echo "    make -f scripts/Makefile docker-logs    Tail logs"
	@echo ""
	@echo "  Training & Evaluation"
	@echo "    make -f scripts/Makefile train          Train DRL agents"
	@echo "    make -f scripts/Makefile evaluate       Evaluate all strategies"
	@echo "    make -f scripts/Makefile figures        Generate figures"
	@echo ""
	@echo "  Analysis"
	@echo "    make -f scripts/Makefile analyze-costs  Transaction cost analysis"
	@echo "    make -f scripts/Makefile analyze-reward Reward ablation study"
	@echo "    make -f scripts/Makefile analyze-regime Market regime analysis"
	@echo ""
	@echo "  API"
	@echo "    make -f scripts/Makefile api            Start API (dev, reload)"
	@echo "    make -f scripts/Makefile api-prod       Start API (production)"
	@echo ""
	@echo "  Cleanup"
	@echo "    make -f scripts/Makefile clean          Remove cache / build artefacts"
	@echo "    make -f scripts/Makefile clean-data     Remove data / model / result files"

# ── Installation ──────────────────────────────────────────────────────────
install:
	pip install -r code/requirements.txt
	pip install -r code/requirements-prod.txt

install-dev: install
	pip install pytest pytest-cov black flake8 mypy

# ── Quality ───────────────────────────────────────────────────────────────
test:
	pytest code/tests/ -v --cov=code --cov-report=html

lint:
	flake8 code/ --max-line-length=100
	mypy code/

format:
	black code/

# ── Docker ────────────────────────────────────────────────────────────────
docker-build:
	$(COMPOSE) build

docker-up:
	$(COMPOSE) up -d
	@echo "Services started!"
	@echo "  API:     http://localhost:8000"
	@echo "  Jupyter: http://localhost:8888"
	@echo "  Grafana: http://localhost:3000"

docker-down:
	$(COMPOSE) down

docker-logs:
	$(COMPOSE) logs -f

docker-restart:
	$(COMPOSE) restart

# ── Training & Evaluation ─────────────────────────────────────────────────
train:
	python code/training/train.py

evaluate:
	python code/evaluation/evaluate.py

figures:
	python code/analysis/figure_generation.py

# ── Analysis ─────────────────────────────────────────────────────────────
analyze-costs:
	python code/analysis/transaction_cost_analysis.py

analyze-reward:
	python code/analysis/reward_ablation.py

analyze-regime:
	python code/analysis/regime_analysis.py

# ── API ──────────────────────────────────────────────────────────────────
api:
	uvicorn code.production.api:app --host 0.0.0.0 --port 8000 --reload

api-prod:
	uvicorn code.production.api:app --host 0.0.0.0 --port 8000 --workers 4

# ── Cleanup ──────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage

clean-data:
	rm -rf data/* models/* results/*

# ── Database ─────────────────────────────────────────────────────────────
db-init:
	docker exec -it rl-portfolio-db psql -U portfolio_user -d portfolio_db \
	    -f /docker-entrypoint-initdb.d/init.sql

db-backup:
	docker exec rl-portfolio-db pg_dump -U portfolio_user portfolio_db \
	    > backup_$(shell date +%Y%m%d_%H%M%S).sql

db-restore:
	@read -p "Enter backup file name: " file; \
	docker exec -i rl-portfolio-db psql -U portfolio_user portfolio_db < $$file

# ── Development helpers ───────────────────────────────────────────────────
notebook:
	jupyter notebook notebooks/

shell:
	docker exec -it rl-portfolio-api bash

# ── All-in-one ────────────────────────────────────────────────────────────
setup: install docker-build docker-up
	@echo "Setup complete! Run 'make -f scripts/Makefile train' to train models."

full-pipeline: train evaluate figures
	@echo "Full pipeline complete! Check results/ directory."
