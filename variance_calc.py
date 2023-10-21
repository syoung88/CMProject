import statistics

# density variance :)
# density values from https://www.usgs.gov/special-topics/water-science-school/science/water-density

density_list = [0.99987, 1, 0.99999, 0.99975, 0.99907, 0.99802, 0.99669, 0.99510, 0.99318, 0.98870, 0.98338, 0.97729, 0.97056, 0.96333, 0.95865]
d = statistics.stdev(density_list) * 1000
print(d)
