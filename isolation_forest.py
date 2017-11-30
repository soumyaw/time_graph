import global_outlier_detection as g_out
from sklearn.ensemble import IsolationForest
import make_networkx_graph as mk_graph
import datetime
import matplotlib.pyplot as plt
import networkx as nx
import operator

clf = IsolationForest()

ONE_DAY = 86400
NUM_DAYS_WEEK = 7
NUM_DAYS_MONTH = 30
NUM_DAYS_YEAR = 365

def main():
	file_name = "data/enron_unix.txt"
	granularity = "daily"
	n = 20
	edges = mk_graph.get_edges(file_name, granularity)
	time_keys = edges.keys()
	time_keys.sort()
	# print time_keys
	# print len(time_keys)
	if granularity == 'daily':
		time_unit = ONE_DAY
	elif granularity == 'weekly':
		time_unit = ONE_DAY*NUM_DAYS_WEEK
	elif granularity == 'monthly':
		time_unit = ONE_DAY*NUM_DAYS_MONTH
	elif granularity == 'yearly':
 		time_unit = ONE_DAY*NUM_DAYS_YEAR
	X = []
	edges_union = []
	graph_prev = nx.MultiGraph()
	for key in time_keys:
		edge_curr = edges[key]
		edges_union.extend(edge_curr)
		graph = nx.MultiGraph()
		graph.add_edges_from(edge_curr)
		X.append(g_out.inp_vectors(graph, graph_prev))
		graph_prev = graph
	clf.fit(X)
	y_pred = clf.decision_function(X)
	plt.plot(time_keys, -y_pred)
	plt.savefig('results/time_vs_neg_score_'+granularity+'.png')
	plt.close()
	combined = sorted(zip(time_keys, y_pred, X), key=operator.itemgetter(1))
	len_proc = len(X)
	print "The top "+str(n)+" anomalies based on global outlier detection:"
	for i in range(n):
		print datetime.datetime.utcfromtimestamp(int(combined[i][0])*time_unit).strftime('%Y-%m-%d')
		print combined[i][1]
		print combined[i][2]
		print '...'
	y_pred_sorted = [comb[1] for comb in combined]
	plt.plot(range(len_proc), y_pred_sorted)
	plt.savefig('results/rank_vs_score_'+granularity+'.png')

if __name__ == "__main__":
	main()