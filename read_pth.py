import numpy as np
import torch

# Open weight files.
pfw = open("mobilenet.weight", "w")

pthfile = r"mobilenet_sgd_68.848.pth"
net = torch.load(pthfile, map_location=torch.device('cpu'))
print (type(net))
print (len(net))

for k in net.keys():
    print(k)
# state_dict epoch arch optimizer best_prec1

# Write pth weights to model.weight
for key,value in net["state_dict"].items():
    print(key,value.size(), value.dtype, sep=" ")
    #print(value)
    print("ndim: ", value.ndim)
    val_array = value.numpy()
    for ele in np.nditer(val_array):
        pfw.write("%e"% ele)

pfw.close()



