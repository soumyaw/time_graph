import networkx as nx

ONE_DAY = 86400
NUM_DAYS_WEEK = 7
NUM_DAYS_MONTH = 30
NUM_DAYS_YEAR = 365

def get_edges(file_name, granularity = 'daily'):
    if granularity == 'daily':
        time_unit = ONE_DAY
    elif granularity == 'weekly':
        time_unit = ONE_DAY*NUM_DAYS_WEEK
    elif granularity == 'monthly':
        time_unit = ONE_DAY*NUM_DAYS_MONTH
    elif granularity == 'yearly':
        time_unit = ONE_DAY*NUM_DAYS_YEAR
    f = open(file_name, 'r')
    edges = {}
    for row in f:
        row = row.strip().split()
        time = int(float(row[2]))/time_unit
        if time not in edges:
            edges[time] = []
        edges[time].append((row[0], row[1]))   
    f.close()
    return edges