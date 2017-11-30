######
# Authors: Christos Faloutsos, Soumya Wadhwa
# Date: Sept-Oct. 2017
# Goal: analysis for time-evolving graphs
######

top:
	python run_global.py
	python isolation_forest.py
	python local_discont.py
clean:
	\rm -f *.pyc

spotless: clean

