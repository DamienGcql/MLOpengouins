
.PHONY: setup
setup :
	@echo "Setting up the development environment..."
	pyenv virtualenv pingouins
	pyenv local pingouins
	pip install -e .
	@echo "✅ Development environment setup complete."
