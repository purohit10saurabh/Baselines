.ONESHELL:
all: clean
	echo `python --version`
	python -W ignore setup.py build_ext --inplace

clean:
	rm -rf build
	rm -rf __pycache__
	rm -rf util.cpp
	rm -rf util.*.so 
