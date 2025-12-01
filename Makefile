
.PHONY: setup
setup :
	pyenv virtualenv pingouins
	pyenv local pingouins
	pip install -r requirements.txt