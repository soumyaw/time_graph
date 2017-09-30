import make_networkx_graph as mk_graph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.algorithms import bipartite
import sys

ONE_DAY = 86400
NUM_DAYS_WEEK = 7
NUM_DAYS_MONTH = 30
NUM_DAYS_YEAR = 365

def plot_measures(func_1, func_2, measure_name, time, file_name=None, time_keys=None, edges=None, granularity='daily'):
	if granularity == 'daily':
		time_unit = ONE_DAY
	elif granularity == 'weekly':
		time_unit = ONE_DAY*NUM_DAYS_WEEK
	elif granularity == 'monthly':
		time_unit = ONE_DAY*NUM_DAYS_MONTH
	elif granularity == 'yearly':
 		time_unit = ONE_DAY*NUM_DAYS_YEAR
	if file_name:
		edges = mk_graph.get_edges(file_name, granularity)
		time_keys = edges.keys()
		time_keys.sort()
	edges_union  = []
	val1 = []
	val2 = []
	for key in time_keys:
		edge_curr = edges[key]
		edges_union.extend(edge_curr)
		if key == time:
			graph = nx.Graph(edges_union)
			graph_nodes = nx.classes.function.nodes(graph)
			val1_ = func_1(graph)
			val2_ = func_2(graph)
			for node in graph_nodes:
				print node
				val1.append(val1_[node])
				print val1[-1]
				val2.append(val2_[node])
				print val2[-1]
			break
	x = np.array(val1)
	y = np.array(val2)
	xmin = x.min()
	xmax = x.max()
	ymin = y.min()
	ymax = y.max()
	extent = [xmin, xmax, ymin, ymax]
	plt.hexbin(x, y, gridsize=50, cmap='hot', extent=extent)
	plt.savefig(measure_name+"_node.png")
	plt.clf()
	plt.hexbin(x, y, gridsize=50, bins='log', cmap='hot', extent=extent)
	plt.savefig(measure_name+"_node_log.png")
	plt.clf()

# plot_measures(nx.classes.function.degree, nx.algorithms.link_analysis.pagerank, "degree and pagerank", 15000, "data/sx-mathoverflow.txt")