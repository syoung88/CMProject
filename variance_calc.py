import pandas as pd
import statistics
import numpy as np
from matplotlib import pyplot as plt

# # using dataset containing surface areas.
#
# aq_df = pd.read_csv("/Users/maddydon/Desktop/1_acquifer_project/CMProject/location-and-extent-of-nzs-aquifers-2015.csv")
# area = aq_df.AREA
#
# area_list = []
# for i in area:
#     area_list += [i]
#
# v = statistics.variance(area_list)

# using dataset containing 'depth to hydrogeoloical basement'.

aq2_df = pd.read_csv("C:/Users/a1exa/Desktop/ENGSCI 263/CM Project/nationally-consistent-hydrogeological-unit-map-2019.csv")
d_max = aq2_df[['base_MAX', 'HUM_type']]
d_max = d_max.loc[d_max['HUM_type'] == 'Aquifer']
d_min = aq2_df[['base_MIN', 'HUM_type']]
d_min = d_min.loc[d_min['HUM_type'] == 'Aquifer']
poly_ar = aq2_df[['poly_area', 'HUM_type']]
poly_ar_filter1 = poly_ar[poly_ar['HUM_type'] == 'Aquifer']
poly_ar_filter2 = poly_ar_filter1[poly_ar_filter1['poly_area'] < 200]


depth_max = []
for j in d_max.base_MAX:
    depth_max += [j]

depth_min = []
for k in d_min.base_MIN:
    depth_min += [k]

poly_area = []
for f in poly_ar_filter2.poly_area:
    poly_area += [f]

figb, (ax2) = plt.subplots(1, 1)
num_bins = 50
ax2.hist(poly_area, num_bins)
plt.show()

a = np.sqrt(statistics.variance(poly_area))
print(a)

# depth_diff = [0] * 409
# vol = [0] * 409
# for i in range(len(depth_max)):
#     depth_diff[i] = depth_max[i] - depth_min[i]
#     vol[i] = depth_diff[i] * poly_area[i]
#
# volume = []
# for i in range(len(vol)):
#     if vol[i] < 6000.0 or vol[i] > 5000.0:
#         volume.append(vol[i])
#
# v = statistics.variance(volume)
# print(v)
