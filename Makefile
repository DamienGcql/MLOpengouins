
.PHONY: setup
setup :
	@echo "Setting up the development environment..."
	pyenv virtualenv pingouins || true
	pyenv local pingouins
	pip install -e .
	@echo "✅ Development environment setup complete."
