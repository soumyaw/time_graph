#####
# Author: Soumya Wadhwa
# Date: Oct 2017
# Goal: target for makefile
#####

import global_outlier_detection as glb_measure
import global_plot as glb_plt

# other measures can be added and plotted (from global_outlier_detection) 

glb_plt.lag_plot_measure(glb_measure.get_num_nodes, "num_nodes", "data/sx-mathoverflow.txt")