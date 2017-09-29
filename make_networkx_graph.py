import networkx as nx

def get_daily_edges(file_name):
    f = open(file_name, 'r')
    users = {}
    prods = {}
    node_no = 0
    edges = {}
    for row in f:
        row = row.strip().split(',')
        if row[0] not in users:
            users[row[0]] = node_no
            user = node_no
            node_no += 1
        else:
            user = users[row[0]]
        if row[1] not in prods:
            prods[row[1]] = node_no
            prod = node_no
            node_no += 1
        else:
            prod = prods[row[1]]
        row[3] = float(row[3])
        if row[3] not in edges:
            edges[row[3]] = []
        edges[row[3]].append((user, prod))
    f.close()
    return edges

def get_weekly_edges(edges=None):
    weekly_edges = {}
    return weekly_edges

def get_monthly_edges(edges=None):
    monthly_edges = {}
    return monthly_edges

def get_yearly_edges(edges=None):
    yearly_edges = {}
    return yearly_edges
