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

def plot_measure(func, measure_name, num=None, file_name=None, time_keys=None, edges=None, granularity='daily'):
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
	measure = {}
	measure_vals = []
	edges_union = []
	for key in time_keys:
		edge_curr = edges[key]
		edges_union.extend(edge_curr)
		graph = nx.MultiGraph()
		graph.add_edges_from(edge_curr)
		if num!=None:
			val = func(graph)[num]
		else:
			val = func(graph)
		measure[key] = val
		# print val
		measure_vals.append(val)
	plt.plot(time_keys, measure_vals)
	plt.savefig("results/"+measure_name+"_"+granularity+".png")
	plt.clf()
	return time_keys, measure

def lag_plot_measure(func, measure_name, num=None, file_name=None, time_keys=None, edges=None, granularity='daily'):
	time_keys, measure = plot_measure(func, measure_name, num, file_name, time_keys, edges, granularity)
	lag_points_x = []
	lag_points_y = []
	if granularity == 'daily':
		time_unit = ONE_DAY
	elif granularity == 'weekly':
		time_unit = ONE_DAY*NUM_DAYS_WEEK
	elif granularity == 'monthly':
		time_unit = ONE_DAY*NUM_DAYS_MONTH
	elif granularity == 'yearly':
		time_unit = ONE_DAY*NUM_DAYS_YEAR
	seen = {}
	for key in time_keys:
		key_next = key + time_unit
		lag_points_x.append(measure[key])
		if key_next not in time_keys:			
			lag_points_y.append(measure[key])
		else:
			lag_points_y.append(measure[key_next])
	x = np.array(lag_points_x)
	y = np.array(lag_points_y)
	xmin = x.min()
	xmax = x.max()
	ymin = y.min()
	ymax = y.max()
	extent = [xmin, xmax, ymin, ymax]
	#plt.hexbin(x, y, gridsize=100, cmap='YlOrRd', extent=extent)
	#plt.savefig(measure_name+"_"+granularity+"_lag.png")
	#plt.clf()
	plt.hexbin(x, y, gridsize=100, bins='log', cmap='YlOrRd', extent=extent)
	plt.savefig("results/"+measure_name+"_"+granularity+"_log_lag.png")
	plt.clf()

# lag_plot_measure(nx.classes.function.number_of_nodes, "num_nodes", "data/sx-mathoverflow.txt")

