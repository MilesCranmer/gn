#!/usr/bin/env python
# coding: utf-8

# Let's load relevant libraries. You will need to install https://github.com/franciscovillaescusa/Pylians3 and put it on your path. You also need the scientific computing stack (numpy, matplotlib, etc.) as well as PyTorch, PyTorch geometric, and tqdm.

# In[4]:


from quijote_gn_nv import *


# This loads a single simulation's graph, with a cutoff of 50 units between nodes. This creates the following histogram of the # of connections per node:

# In[5]:


graph_data = load_graph_data(realization=0, cutoff=20)  # make cutoff smaller for code to run faster (e.g., 20)
#initial_mask = graph_data['graph'].y[:, 1].clone()


# Let's create a graph network with 500 hidden nodes (so the hidden layer matrix multiplications are 500x500) and message dimension of 100.

# In[6]:

ogn = create_graph_network(hidden=275, msg_dim=5)


# Let's run the training on this realization, trying to predict the dark matter overdensity, and time it.

# In[ ]:


out_loss = do_training(
    ogn, graph_data['graph'],
    total_epochs=50, batch_per_epoch=15, batch=175,
    weight_decay=3.5882522951978176e-07, l1=0.0001468270725905777
);


# In[ ]:




