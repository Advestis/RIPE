mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))


help:
	@echo "Run :"
	@echo "  - make install to install the program"
	@echo "  - make doc to compile the doc"
	@echo "  - make a_command to run python setup.py 'a_command'"

.PHONY: help Makefile

doc:
	@echo "RIPE does not support documentation. Skipping."

%: Makefile
	@echo "Running python setup.py "$@"..."
	@if [ -f apt-requirements.txt ] ; then if command -v sudo > /dev/null ; then sudo apt-get install -y $(grep -vE "^\s*#" apt-requirements.txt  | tr "\n" " ") else apt-et install -y $(grep -vE "^\s*#" apt-requirements.txt  | tr "\n" " ") ; fi ; fi

	@if [ -f gspip-requirements.txt ] ; then if command -v gspip > /dev/null ; then gspip --upgrade install $(grep -vE "^\s*#" gspip-requirements.txt  | tr "\n" " ") else git clone https://github.com/Advestis/gspip && gspip/gspip.sh --upgrade install $(grep -vE "^\s*#" gspip-requirements.txt  | tr "\n" " ") && rm -rf gspip ; fi ; fi

	@pip3 uninstall ripe-algorithm -y
	@pip3 install setuptools
	@python setup.py $@
	@if [ -d "dist" ] && [ $@ != "sdist" ] ; then rm -r dist ; fi
	@if [ -d "build" ] ; then rm -r build ; fi
	@if [ -d "ripe_algorithm.egg-info" ] ; then rm -r ripe_algorithm.egg-info ; fi
