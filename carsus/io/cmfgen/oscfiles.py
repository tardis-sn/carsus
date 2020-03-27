
import requests 
import os 
from tqdm import tqdm
import re
import tarfile
from pprint import pprint
import pandas as pd
import numpy as np


working_directory=os.getcwd()


#pass header as True to get header information
#pass download as False to disable download
def download_database(download=True,header=False) :
  url="http://kookaburra.phyast.pitt.edu/hillier/cmfgen_files/atomic_data_15nov16.tar.gz"
  h = requests.head(url, allow_redirects=True)
  header = h.headers
  filelength=int(header["Content-Length"])

  if header :
    
    print("Content Type : ",header["content-type"])
    print("Last Modified :",header["Last-Modified"])
    print("Todays Date :",header["Date"])
    print("File Size :",filelength/10e5,"MB")


  if download : 
    filename=url.split("/")[-1]
    if os.path.exists(filename) :
      return print("File already exists .")
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        pbar = tqdm(total=int(filelength/1024))
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:                   # filter out keep-alive new chunks
              pbar.update ()
              f.write(chunk)





def parse_header(output,start=0,end=50) :   #start and end can be changed if a header is expcted to be beyond default range
  

  pattern="![A-Za-z]+.+"
  p=re.compile(pattern)

  keys={}
  count=[]
  for x in range(start,min(len(output),end)) :
    temp=p.findall(output[x])
    if temp != [] :
      value=output[x].split(" ")[0]
      keys[temp[0][1:]]=value
      count.append(x)
  
  return keys,count[0],count[-1]


class CMFGENEnergyLevelsParser():

  '''
  modelled on the base parser class , yet to be inherited 
  
  attributes : 
  meta = returns meta header data of the file
  load method =loads the compressed database and searches for the passed file (extraaction of file is not required )
  fname= represents the filename passed at class insantiation 
  base = represents a panda dataframe composed of the parsed data 
  
  Note : 
  This parser has been re written and the logic has been changed and the find row method been deprecated to optimize the code 
  to be excecuted within one pass of the file .
  
  Important Points :
  The to_hdf() function will produce the output in your current working directory .
  To specfiy a path where the output has to be taken pass the specifc file path to the fname parameter in the function call.
  '''    
  def __init__ (self,fname) :

    self.load(fname)

  def load(self,fname) :
    base_filename="atomic_data_15nov16.tar.gz"
    t = tarfile.open(base_filename,'r:gz')
    if not fname.startswith("a") :
      temp=fname.find('a')
      fname=fname[temp:]



    file=t.extractfile(fname)
    #output will contain all of the file content line by line 
    output=list(map(lambda x: x.strip().decode("utf-8"),file.readlines()))
    meta,cstart,skip=parse_header(output)
    new_output=list(map(lambda x: x.split(),output))
    columns=[x for x in new_output[cstart-3] if x!=" "]
 
    n=int(meta['Number of energy levels'])
   
    df = pd.DataFrame(new_output[skip+2:skip+n+2],columns=columns,index=range(0,n))
 

    
    self.meta=meta
    self.base=df
    self.fname=fname
    self.columns=columns

  def to_hdf(self, key='/energy_levels',fname=working_directory):
    if not self.base.empty:
      with pd.HDFStore('{}.h5'.format(fname), 'a') as f:
          f.put(key, self.base)
          f.get_storer(key).attrs.metadata = self.meta


class CMFGENOscillatorStrengthsParser():


  '''
  modelled on the base parser class , yet to be inherited 
  
  attributes : 
  meta = returns meta header data of the file
  load method =loads the compressed database and searches for the passed file (extraaction of file is not required )
  fname= represents the filename passed at class insantiation 
  base = represents a panda dataframe composed of the parsed data 
  
  Note : 
  This parser has been re written and the logic has been changed and the find row method been deprecated to optimize the code 
  to be excecuted within one pass of the file .
  
  Important Points :
  The to_hdf() function will produce the output in your current working directory .
  To specfiy a path where the output has to be taken pass the specifc file path to the fname parameter in the function call.
  '''
  def __init__ (self,fname) :

    self.load(fname)

  def load(self,fname) :
    base_filename="atomic_data_15nov16.tar.gz"
    t = tarfile.open(base_filename,'r:gz')
    if not fname.startswith("a") :
      temp=fname.find('a')
      fname=fname[temp:]
    file=t.extractfile(fname)
    #output will contain all of the file content line by line 
    output=list(map(lambda x: x.strip().decode("utf-8"),file.readlines()))
    meta,cstart,skip=parse_header(output)
    new_output=list(map(lambda x: x.split(),output))
    columns1 = ['State A', 'State B', 'f', 'A','Lam(A)', 'i', 'j', 'Lam(obs)', '% Acc']
    columns2=['State A', 'State B', 'f', 'A','Lam(A)', 'i', 'j', 'Transition Number']
    
    
    n=int(meta['Number of energy levels'])
    m=int(meta['Number of transitions'])
    cstart=0

    for start in range(skip+n,skip+n+100) :
      if len(new_output[start])<1 :
        continue

      if new_output[start][0]=="Transition" :

        cstart=start
        break

    for x in new_output[cstart+2:cstart+2+m+1]:
      i=x[0].find("-")
      try :
        if i !=-1 :
          x.insert(1,x[0][i+1:])
          x[0]=x[0][:i]
          
        else :
          x[1]=x[1][1:]
    
        if x[5].endswith("-") :
          x[5]=x[5][:-1]
        if x[8]=="|" :
          x[7]=np.nan
        else :
          x[7]=x[8]
        if len(x)==9+1 :
          if x[9]== "|" :
            x[8]=np.nan
          else :
            x[8]=x[9]
          x.pop(9)
        elif len(x)==10 +1:
          if x[9]=="|" :
            x[8]=np.nan

          else :
            x[8]=x[9]
          x.pop(10)
          x.pop(9)
          
        elif len(x)==11+1 :
          x[7]=x[8]
          x[8]=x[10]
          x.pop(11)
          
          x.pop(10)
          x.pop(9)
      except IndexError :
        pass

    if len(new_output[skip+n+1+13])==9 :
      columns=columns1
    else :
      columns=columns2
    df = pd.DataFrame(new_output[cstart+2:cstart+2+m+1],columns=columns,index=range(0,m))
 

    
    self.meta=meta
    self.base=df
    self.fname=fname
    self.columns=columns


  def to_hdf(self, key='/oscillator_strengths',fname=working_directory):
    if not self.base.empty:
      with pd.HDFStore('{}.h5'.format(fname), 'a') as f:
          f.put(key, self.base)
          f.get_storer(key).attrs.metadata = self.meta


