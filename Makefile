convolve: clean
	python setup.py build_ext --inplace

clean:
	rm -f *.so
