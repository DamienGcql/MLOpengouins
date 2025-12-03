
.PHONY: setup
setup :
	@echo "Setting up the development environment..."
	pyenv virtualenv pingouins || true
	pyenv local pingouins
	pip install -e .
	@echo "✅ Development environment setup complete."

show_config :
	python -m pengouins.config

launch_mlflow :
	@echo "Launching MLflow tracking server..."
	mlflow ui --port $${MLFLOW_LOCAL_PORT:-8888} 
	@echo "✅ MLflow tracking server launched on port $${MLFLOW_LOCAL_PORT:-8888}."
