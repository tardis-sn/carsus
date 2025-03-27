#!/usr/bin/env python
# coding: utf-8

# # 🚀 GSoC 2025: Metadata for Atomic Data in Carsus
# 
# This notebook demonstrates the first objective of the GSoC 2025 project proposal for the Carsus project:  
# **Adding metadata to Carsus atomic data outputs.**
# 
# We simulate a Carsus-like `levels` table, attach metadata including units, git commit, DOI, and citation info, and export everything into a structured HDF5 file.
# 

# In[7]:


# 📦 Install core dependencies (Colab)
get_ipython().system('pip install git+https://github.com/tardis-sn/carsus.git')
get_ipython().system('pip install gitpython uncertainties')


# In[8]:


import pandas as pd
import subprocess
from datetime import datetime
import os
from pathlib import Path


# In[2]:


# Simulate a Carsus-like levels DataFrame
levels_df = pd.DataFrame({
    "atomic_number": [1, 1, 2, 2],
    "ion_charge": [0, 0, 1, 1],
    "level_index": [0, 1, 0, 1],
    "energy": [0.0, 10.2, 0.0, 20.6],
    "j": [2, 8, 1, 3],
    "label": ["1s", "2s", "1s", "2s"],
    "method": ["meas"]*4,
    "priority": [10]*4
})
levels_df["reference"] = "Kurucz GFALL"
levels_df.head()


# In[3]:


# Function to generate metadata DataFrame
def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except:
        return "unknown"

metadata_df = pd.DataFrame({
    "data_source": ["https://doi.org/10.1086/313149"],
    "units": ["eV"],
    "generated_on": [datetime.now().isoformat()],
    "git_commit": [get_git_commit()],
    "notes": ["Energy levels for H and He from Kurucz GFALL"]
})
metadata_df


# In[4]:


# Citation table for A_ij and Υ_ij
citation_df = pd.DataFrame({
    "Ref. A_ij": [
        "Bautista et al. (2015)",
        "Quinet (1996)",
        "Storey et al. (2016)",
        "Cassidy et al. (2016)",
        "Fivet et al. (2016)"
    ],
    "Ref. Υ_ij": [
        "Bautista et al. (2015)",
        "Zhang (1996)",
        "Storey et al. (2016)",
        "Cassidy et al. (2010)",
        "Watts & Burke (1998)"
    ]
})
citation_df


# In[5]:


# Save levels, metadata, and citations to HDF5
output_path = "carsus_with_metadata.h5"

with pd.HDFStore(output_path) as store:
    store.put("levels", levels_df)
    store.put("levels_metadata", metadata_df)
    store.put("levels_citations", citation_df)

print(f"✅ Data saved to {output_path}")


# In[6]:


# Load and verify contents
with pd.HDFStore(output_path) as store:
    print("Available datasets:")
    print(store.keys())
    print("\nMetadata preview:")
    display(store["levels_metadata"])


# ## ✅ Summary
# 
# This notebook demonstrates:
# - A Carsus-style atomic `levels` table
# - Embedded metadata with source, units, timestamp, git commit
# - Citation references (A<sub>ij</sub>, Υ<sub>ij</sub>)
# - Exported HDF5 file with all content included
# 
# This fulfills the **first objective** for Carsus metadata integration.
# 
