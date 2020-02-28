import numpy as np
import torch

# Open weight files.
pfw = open("mobilenet.weight", "w")
pbfw = open("mobilenetbin.weight", "w+b")

pthfile = r"mobilenet_sgd_68.848.pth"
net = torch.load(pthfile, map_location=torch.device('cpu'))
print (type(net))
print (len(net))

for k in net.keys():
    print(k)
# state_dict epoch arch optimizer best_prec1
#for param_tensor in net["state_dict"]:
#    print(param_tensor)
#print(net["state_dict"].keys())

#for var_name in net["optimizer"]:
#    print(var_name)

# Write pth weights to model.weight
for key,value in net["state_dict"].items():
    print(key, value.size(), value.ndim, value.dtype, sep=" ")
    #print(value)
    #print("ndim: ", value.ndim)
    pfw.write(key)
    pfw.write(" ")
    mylist = list(value.size())
    pfw.write(str(mylist).strip('[]'))
    pfw.write(" ")
    pfw.write(str(value.ndim))
    pfw.write(" ")
    pfw.write(str(value.dtype))
    pfw.write(str(value.size()))
    val_array = value.numpy()
    pfw.write("\n")
    i = 0
    for ele in np.nditer(val_array):
        if key == "module.model.12.3.weight":
            if i < 524228:
                pfw.write("%e "% ele)
                pbfw.write(ele.tobytes())
                i = i + 1
        else:
            pfw.write("%e "% ele)
            pbfw.write(ele.tobytes())
            i = i + 1
    pfw.write("num = %d \n" % i)

pfw.close()
pbfw.close()



