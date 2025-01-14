conditional = False

if conditional:
    from ceinsum import EinsumNetwork
else:
    from einsum import EinsumNetwork

#exponential_family = EinsumNetwork.BinomialArray
exponential_family = EinsumNetwork.CategoricalArray
#exponential_family = EinsumNetwork.NormalArray

exponential_family_args = None
if exponential_family == EinsumNetwork.BinomialArray:
    exponential_family_args = {'N': 255}
if exponential_family == EinsumNetwork.CategoricalArray:
    exponential_family_args = {'K': 256}
if exponential_family == EinsumNetwork.NormalArray:
    exponential_family_args = {'min_var': 1e-6, 'max_var': 0.01}

classes = [7]
# classes = [2, 3, 5, 7]
# classes = None

K = 50

structure = 'poon-domingos'
#structure = 'binary-trees'

# 'poon-domingos'
#pd_num_pieces = [8]
#pd_num_pieces = [8]
pd_num_pieces = [8]
width = 32
height = 32
num_vars = width*height
num_dims = 3
patch_size = (height, width)

# 'binary-trees'
depth = 4
num_repetitions = 20

num_epochs = 10
batch_size = 64
online_em_frequency = 10
online_em_stepsize = 0.1
num_clients = 1

checkpoint_freq = 2

# Server config
communication_rounds = 1
reuse_trained = False

# Client config
port = '12005'

# Devices
devices = [5]
num_processes = 1
