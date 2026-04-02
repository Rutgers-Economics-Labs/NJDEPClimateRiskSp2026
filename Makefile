# NJDEP Climate Risk & Resilience Project - Master Makefile

SHELL := /bin/bash

.PHONY: help install clean process-all dashboard-backend dashboard-frontend run-dashboard dev stop-dashboard

help:
	@echo "NJDEP Project Automation"
	@echo "------------------------"
	@echo "make install           - Install all Python and Node dependencies"
	@echo "make process-all       - Run the full data pipeline (Boundaries -> Census -> FEMA -> WRDS)"
	@echo "make dev               - Launch backend and frontend together for local development"
	@echo "make run-dashboard     - Alias for make dev"
	@echo "make clean             - Remove temporary files and pycache"

install:
	pip install pandas geopandas thefuzz rapidfuzz fastapi uvicorn requests
	cd dashboard/frontend && npm install

process-all:
	@echo "Step 1: Running core processing pipeline (Boundaries, Census, Finance)..."
	python3 data_processing/run_all.py
	@echo "Step 2: Building Master Municipality Characteristics Panel..."
	python3 data_processing/build_master_panel.py
	@echo "Step 3: Extracting NJ Bond Data from WRDS (This may take a few minutes)..."
	python3 data_processing/process_wrds_data.py
	@echo "Pipeline Complete."

dashboard-backend:
	python3 -m uvicorn dashboard.backend.main:app --reload --host 127.0.0.1 --port 8000

dashboard-frontend:
	cd dashboard/frontend && npm run dev -- --host 127.0.0.1

dev:
	@echo "Launching dashboard..."
	@echo "Frontend: http://127.0.0.1:5173"
	@echo "Backend:  http://127.0.0.1:8000"
	@backend_pid=""; \
	frontend_pid=""; \
	trap 'if [ -n "$$backend_pid" ]; then kill $$backend_pid 2>/dev/null || true; fi; if [ -n "$$frontend_pid" ]; then kill $$frontend_pid 2>/dev/null || true; fi' EXIT INT TERM; \
	python3 -m uvicorn dashboard.backend.main:app --reload --host 127.0.0.1 --port 8000 & \
	backend_pid=$$!; \
	(cd dashboard/frontend && npm run dev -- --host 127.0.0.1) & \
	frontend_pid=$$!; \
	wait

run-dashboard: dev

stop-dashboard:
	@echo "Stopping dashboard services on ports 8000 and 5173..."
	-lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	-lsof -ti:5173 | xargs kill -9 2>/dev/null || true


clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf data/data_cleaned/*.csv
