from sklearn.ensemble import IsolationForest
import make_networkx_graph as mk_graph
from networkx.algorithms import approximation
import operator
import networkx as nx

clf = IsolationForest()

def get_num_nodes(G):
	return nx.classes.function.number_of_nodes(G)

def get_num_edges(G):
	return nx.classes.function.number_of_edges(G)

def get_gcc_size_and_cc_num(G):
	gcc = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
	G0 = gcc[0]
	return (len(G0), len(gcc))

def get_avg_clustering_coeff_approx(G):
	return approximation.clustering_coefficient.average_clustering(G)	

def get_first_eig_val_adj(G):
	return nx.linalg.spectrum.adjacency_spectrum(G)[0]
	
def get_first_eig_val_lap(G):
	return nx.linalg.spectrum.laplacian_spectrum(G)[0]

def get_pairwise_dist_tiles(G):
	path_lens = nx.algorithms.shortest_paths.weighted.all_pairs_dijkstra_path_length(G)
	dict_lens = {}
	for src in path_lens.keys():
		for dest in path_lens[src]:
			dict_lens[(src, dest)] = path_lens[src][dest]
	sorted_lens = sorted(dict_lens.items(), key=operator.itemgetter(1))
	length = len(sorted_lens)
	return [sorted_lens[i*length/10][1] for i in range(1, 10)]

def inp_vectors(G):
	num_nodes = get_num_nodes(G)
	num_edges = get_num_edges(G)
	gcc_size_and_cc_num = get_gcc_size_and_cc_num(G)
	gcc_size = gcc_size_and_cc_num[0]
	cc_num = gcc_size_and_cc_num[1]
	clus_coeff = get_avg_clustering_coeff_approx(G)
	first_eig_adj = get_first_eig_val_adj(G)
	first_eig_lap = get_first_eig_val_lap(G)
	pw_dist = get_pairwise_dist_tiles(G)
	ten_pw_dist = pw_dist[0]
	ninety_pw_dist = pw_dist[8]
	return [num_nodes, num_edges, gcc_size, cc_num, clus_coeff, first_eig_lap, ten_pw_dist, ninety_pw_dist]

def main():
	file_name = "data/sx-mathoverflow.txt"
	granularity = "monthly"
	edges = mk_graph.get_edges(file_name, granularity)
	time_keys = edges.keys()
	time_keys.sort()
	X = []
	edges_union = []
	for key in time_keys:
		edge_curr = edges[key]
		edges_union.extend(edge_curr)
		graph = nx.Graph(edges_union)
		X.append(inp_vectors(graph))
		print X[-1]
	clf.fit(X)
	y_pred = clf.predict(X)
	print y_pred

if __name__ == "__main__":
	main()
