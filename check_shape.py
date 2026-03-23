import numpy as np
m = np.load("output_v5b_half/596x199/group_1/pos_mean.npy")
s = np.load("output_v5b_half/596x199/group_1/pos_std.npy")
print("mean:", m.shape)
print("std:", s.shape)
print("spatial:", np.load("output_v5b_half/596x199/group_1/spatial_size.npy"))
