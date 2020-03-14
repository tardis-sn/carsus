#!/usr/bin/env python
# coding: utf-8

# In[21]:


import os
import warnings
import sys
import numpy as np
import pandas as pd
# print(sys.path)


# In[3]:


from carsus.io.cmfgen import (hdf_dump, CMFGENEnergyLevelsParser, CMFGENOscillatorStrengthsParser,
                              CMFGENCollisionalDataParser, CMFGENPhotoionizationCrossSectionParser)


# In[17]:
"""Getting the CMFGEN dataset directory, assuming the dataset is located in the parent dir"""

os.environ['CMFGEN_DIR'] = '../CMFGEN/'
cmfgen_dir = os.getenv('CMFGEN_DIR')
chunk_size = 10
# os.listdir(cmfgen_dir)
print(cmfgen_dir)


# In[14]:

"""Setting file patterns to be matched"""
osc_patterns = ['osc', 'OSC', 'Osc']


# In[34]:


hdf_dump(cmfgen_dir, osc_patterns, CMFGENEnergyLevelsParser(), chunk_size)


# In[16]:


ignore_patterns = ['ERROR_CHK', 'hmi_osc']
hdf_dump(cmfgen_dir, osc_patterns, CMFGENOscillatorStrengthsParser(),
         chunk_size, ignore_patterns)
# In[45]:


# In[18]:
"""Defining utility functions, help in finding specific rows"""


def search_header(file, string):
    with open(file) as File:
        for line in File:
            if string in line:
                break

    n = int(line.split()[0])
    return n


# In[19]:


def find_row(file, string):
    with open(file) as File:
        n = 0
        for line in File:
            n += 1
            if string in line:
                break
    return (n - 1)


# In[22]:
"""Setting file path and fine-tuning arguments, assuming the dataset is located in the parent dir"""

file = '../CMFGEN/atomic/SIL/II/16sep15/si2_osc_kurucz'
args = {}
args['header'] = None
args['delim_whitespace'] = True

args['nrows'] = search_header(file, "Number of energy levels")
args['skiprows'] = find_row(file, "0.000")

"""Getting Energy Levels Header"""

energy_levels = pd.read_csv(file, **args)
energy_levels.columns = ['Energy Level', 'g',
                         'E(cm^-1)', '10^15 Hz', 'eV', 'Lam(A)', 'ID', 'ARAD', 'C4', 'C6']


# In[23]:

"""Printing Energy Levels Header"""


print(energy_levels.to_string())


# In[30]:
"""Printing oscillator strengths header in a naive way"""


args['nrows'] = search_header(file, "Number of transitions")
args['skiprows'] = find_row(file, "Transition") + 1

oss = pd.read_csv(file, **args)
print(oss.head().to_string(), "\n", oss.tail().to_string())


# In[31]:

"""Printing oscillator strengths header using fixed column widths. Method not easily extensible to other files."""
widths = [(0, 44), (49, 59), (61, 71), (74, 83), (87, 94)]
oss = pd.read_fwf(file, colspecs=widths, **args)
oss.columns = ['Transition', 'f', 'A', 'Lam(A)', 'i-j']  # ,'Lam(obs)','% Acc']
oss['Lam(obs)'] = np.nan
oss['% Acc'] = np.nan
print(oss.iloc[[1, 4195], :].to_string())


# In[33]:

"""Printing oscillator strengths header using regex. Method should work for most files with minimal changes."""


args['delim_whitespace'] = False
args['sep'] = '(?<=[^E])-(?:[ ]{1,})?|(?<!-)[ ]{2,}[-,\|]?'
oscillator_strengths = pd.read_csv(file, **args)
oscillator_strengths.columns = ['Initial', 'Final',
                                'f', 'A', 'Lam(A)', 'i', 'j', 'Lam(obs)', '%Acc', '?']
print(oscillator_strengths.drop(columns='?').head().to_string(),
      '\n', oscillator_strengths.drop(columns='?').tail().to_string())


# In[ ]:
