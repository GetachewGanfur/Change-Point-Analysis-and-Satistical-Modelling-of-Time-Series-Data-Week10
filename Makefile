# Makefile for Brent Oil Price Change Point Analysis Project

.PHONY: help install setup clean test run-backend run-frontend run-notebooks lint format

# Default target
help:
	@echo "Available commands:"
	@echo "  help          - Show this help message"
	@echo "  install       - Install all dependencies"
	@echo "  setup         - Set up the project (create directories, env file)"
	@echo "  clean         - Clean temporary files and caches"
	@echo "  test          - Run tests"
	@echo "  run-backend   - Start the Flask backend"
	@echo "  run-frontend  - Start the React frontend"
	@echo "  run-notebooks - Start Jupyter Lab"
	@echo "  lint          - Run linting on Python code"
	@echo "  format        - Format Python code with black"

# Install dependencies
install:
	pip install -r requirements.txt
	cd dashboard/frontend && npm install

# Project setup
setup:
	mkdir -p data/raw data/processed data/events logs results
	cp .env.example .env || echo "Environment file already exists"
	@echo "Setup complete! Edit .env file if needed."

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	cd dashboard/frontend && rm -rf build/
	@echo "Cleaned temporary files"

# Run tests
test:
	python -m pytest src/ -v

# Start Flask backend
run-backend:
	cd dashboard/backend && python app.py

# Start React frontend
run-frontend:
	cd dashboard/frontend && npm start

# Start Jupyter Lab
run-notebooks:
	jupyter lab

# Lint Python code
lint:
	flake8 src/ dashboard/backend/
	@echo "Linting complete"

# Format Python code
format:
	black src/ dashboard/backend/
	@echo "Code formatting complete"

# Development setup (all at once)
dev-setup: setup install
	@echo "Development setup complete!"
	@echo "Next steps:"
	@echo "1. Edit .env file if needed"
	@echo "2. Run 'make run-backend' in one terminal"
	@echo "3. Run 'make run-frontend' in another terminal"
	@echo "4. Open http://localhost:3000 in your browser"