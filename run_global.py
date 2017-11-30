#####
# Author: Soumya Wadhwa
# Date: Oct 2017
# Goal: target for makefile
#####

import global_outlier_detection as glb_measure
import global_plot as glb_plt

time_gran = 'daily'

glb_plt.lag_plot_measure(glb_measure.get_num_nodes, "num_nodes", file_name="data/enron_unix.txt", granularity=time_gran)

glb_plt.lag_plot_measure(glb_measure.get_num_edges, "num_edges", file_name="data/enron_unix.txt", granularity=time_gran)

glb_plt.lag_plot_measure(glb_measure.get_min_max_degree, "min_degree", num=0, file_name="data/enron_unix.txt", granularity=time_gran)

glb_plt.lag_plot_measure(glb_measure.get_min_max_degree, "max_degree", num=1, file_name="data/enron_unix.txt", granularity=time_gran)

glb_plt.lag_plot_measure(glb_measure.get_gcc_size_and_cc_num, "gcc_size", num=0, file_name="data/enron_unix.txt", granularity=time_gran)

glb_plt.lag_plot_measure(glb_measure.get_gcc_size_and_cc_num, "num_conn_comp", num=1, file_name="data/enron_unix.txt", granularity=time_gran)

glb_plt.lag_plot_measure(glb_measure.get_core_num, "core_num", file_name="data/enron_unix.txt", granularity=time_gran)
