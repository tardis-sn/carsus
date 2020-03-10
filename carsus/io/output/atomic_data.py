"""
=====================
TARDIS Atomic_data module
=====================
created on Feb 29, 2020
"""

import numpy as np
import pandas as pd
import zipfile, urllib.request, shutil
from os import path
import os
import tarfile
import re

class Atomic_Data: 
    
    """ Will serve as base class for all atomic data contained in 
    'si2_osc_kurucz' in order to provide consistent Tabular data.
    
    Class Members
        ----------
        data : dataframe
            raw unprocessed data
            
        AtomicLevels : dataframe
            Contains Energy Levels and statistical weights for Si II
            
        Transitions : dataframe
            Oscillator strengths for LLL 
            
    Example to use
         ----------
         
         x = Atomic_Data()
         
         print ( x.AtomicLevels )
         
         """
    
    
    url = 'http://kookaburra.phyast.pitt.edu/hillier/cmfgen_files/atomic_data_15nov16.tar.gz'
    file_name = 'atomic.tar.gz'
    
    if(path.exists(file_name) or path.exists('cmf_doc')):
        pass
    else:
        with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            if file_name.endswith("tar.gz"):
                tar = tarfile.open(file_name, "r:gz")
                tar.extractall()
                tar.close()
                os.remove(file_name)
            elif file_name.endswith("tar"):
                tar = tarfile.open(file_name, "r:")
                tar.extractall()
                tar.close()
                os.remove(file_name)
                
    file = open('atomic/SIL/II/16sep15/si2_osc_kurucz' , 'r') 
    data = file.readlines()
    start = 0
    end = 0
    t = []
    counter = -1
    words1 = len(data[0].split())
    words2 = 0
    
    for i in data:
        words2 = len(i.split()) 
        
        if abs(words1 - words2) > 2 :
        
            if end - start > 5:
                t.append([start,end])
            
            start = counter+1
            end = counter+1
        
        else:
            end = end + 1
        
        words1 = words2
        counter += 1
    
    t.append([start,counter])
    
    AtomicLevels = pd.DataFrame( columns = ["State", "g" ,"E(cm^-1)", "10^15 Hz", "eV", "Lam(A)","ID","ARAD" ,"C4" ,"C6"] )
    
    for i in range(t[0][0], t[0][1]+1):
        x = re.findall("[a-zA-Z0-9_\/\[\-\+.\)\()]+]*", data[i])
        y=[]
     
        for j in x:
            y.append(re.sub("^-|-$", "" , j))
        
        ydf = pd.DataFrame( columns = ["State", "g" ,"E(cm^-1)", "10^15 Hz", "eV", "Lam(A)","ID","ARAD" ,"C4" ,"C6"] ,data = [y])
        AtomicLevels = AtomicLevels.append(ydf, ignore_index = True)
        
    Transitions = pd.DataFrame(columns = ["Transition from","Transition to" ,"f","A","Lam(A)","i","j"])
    
    for i in range(t[1][0], t[1][1]+1):
        x = re.findall("[a-zA-Z0-9_\/\[\-\+.\)\()]+]*", data[i])
        y=[]
      
        for j in x:
            y.append(re.sub("^-|-$", "" , j))
        
        ydf = pd.DataFrame(columns = ["Transition from","Transition to" ,"f","A","Lam(A)","i","j"], data = [y])
        Transitions = Transitions.append(ydf, ignore_index = True)
