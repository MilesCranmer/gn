#!/usr/bin/env python
# coding: utf-8

# Let's load relevant libraries. You will need to install https://github.com/franciscovillaescusa/Pylians3 and put it on your path. You also need the scientific computing stack (numpy, matplotlib, etc.) as well as PyTorch, PyTorch geometric, and tqdm.

# In[4]:


from quijote_gn_nv import *


# This loads a single simulation's graph, with a cutoff of 50 units between nodes. This creates the following histogram of the # of connections per node:

# In[5]:


graph_data = load_graph_data(realization=0, cutoff=10)  # make cutoff smaller for code to run faster (e.g., 20)
#initial_mask = graph_data['graph'].y[:, 1].clone()


# Let's create a graph network with 500 hidden nodes (so the hidden layer matrix multiplications are 500x500) and message dimension of 100.

# In[6]:

ogn = create_graph_network(hidden=512, msg_dim=64)


# Let's run the training on this realization, trying to predict the dark matter overdensity, and time it.

# In[ ]:


out_loss = do_training(
    ogn, graph_data['graph'],
    total_epochs=20, batch_per_epoch=150, batch=64
);


# In[ ]:




