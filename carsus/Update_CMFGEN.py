"""
=====================
TARDIS UPDATE_CMFGEN module
=====================
created on Mar 10, 2020
"""

from bs4 import BeautifulSoup
import requests
import csv
import pandas as pd
from os import path
import os
import numpy as np
import urllib.request, shutil
import tarfile
import re
import h5py
from carsus.parsers import *


class UPDATE_CMFGEN():
    
    """     
    Description
    ----------
    hidden_folder : String
        It is the name of the folder which is hidden and where all the data 
        is stored. The data includes the extracted file, log file and HDF5 
        files.          
         
    """
    
    hidden_folder = ".CMFGEN"
    

    def __init__(self, *args, **kwargs):
        if args:    
            if(args[0][0]=='.' and args[0]!='.'):
                self.hidden_folder = args[0] 
            else:
                print("The given folder was not hidden.")
                print("The default folder will be used.")
        
        if path.exists(self.hidden_folder):
            pass
        else:
            os.mkdir(self.hidden_folder)
                
                
    def get_links(self, url = "http://kookaburra.phyast.pitt.edu/hillier/web/CMFGEN.htm"):
        """
        Fetches all links and also update the log Table.
        
        Parameters
        ----------
        url (Optional): String
            url where the links are present
        Returns
        ----------
        None
        """
        html_content = requests.get(url).text
        if url != "http://kookaburra.phyast.pitt.edu/hillier/web/CMFGEN.htm":
            first = "https://my-static-site-exampl1601.herokuapp.com/"
        else:
            first = "http://kookaburra.phyast.pitt.edu/hillier/"
        
        soup = BeautifulSoup(html_content, "lxml")
        if path.exists(self.hidden_folder + "/CMFGEN.txt"):
            file = open( self.hidden_folder + "/CMFGEN.txt" , "r")
        else:
            file = open( self.hidden_folder + "/CMFGEN.txt" , "w")
            file.close()
            file = open( self.hidden_folder + "/CMFGEN.txt" , "r")
            
        file_new = open( self.hidden_folder + "/CMFGEN2.txt" ,"w")
        
        for link, date in zip (soup.find_all("a")[2:], soup.find_all("dd")[:]):
        
            new_text = link.text
            new_href = link.get("href")
            
            if url == "http://kookaburra.phyast.pitt.edu/hillier/web/CMFGEN.htm":
                if "http" not in new_href:
                    new_href = first + new_href.split("../")[1]
            else:
                new_href = first + new_href
            new_date = date.text[1:]
            
            if os.stat(self.hidden_folder + "/CMFGEN.txt").st_size == 0:
                prev_date = ""
                prev_text = ""
                prev_href = ""
                prev_version = ":0"
            
            else:
                prev_text = file.readline()
                prev_href = file.readline()
                prev_date = file.readline()
                prev_version = file.readline()
            
            if prev_date != new_date+"\n":
                file_new.write("Inner Text: {}".format(new_text)+"\n")
                file_new.write(new_href)
                file_new.write(date.text+"\n")
                file_new.write("Version:"+str(int(prev_version.split(":")[1])+1))
                file_new.write("\n")
            
            else:
                file_new.write(prev_text)
                file_new.write(prev_href)
                file_new.write(prev_date)
                file_new.write(prev_version)

        file_new.close()
        file.close()
        os.remove( self.hidden_folder + "/CMFGEN.txt")
        os.rename( self.hidden_folder + "/CMFGEN2.txt" , self.hidden_folder + "/CMFGEN.txt")
    
    
    def file_With_Extension(self, url, name):
        """
        Return complete file name whose initial name is given and extension is
        to be determined from url where this file is uploaded         
        
        Parameters
        ----------
        url: String
            url where the file is located
        name: String
            name with which you want to store the file on the local machine
        Returns
        ----------
        String
            complete file name(along extension)
        """
        
        return_file = name
        
        if url.endswith("tar.gz"):
            return_file = return_file +'.tar.gz'
        elif url.endswith("tar"):
            return_file = return_file + '.tar'
            
        return return_file
        
    
    def download_data (self, url, name):
        """
        Downloads data from a url and stores with a particular name
        (but upgraded version).
        
        Parameters
        ----------
        url: String
            url where the file is located
        name: String
            name with which you want to store the file on the local machine
        Returns
        ----------
        None
        """
        antepenult_ver = name.split("@")[0] + "@" + str(int(name.split("@")[1])-2)
        antepenult_ver = self.hidden_folder + "/"+ antepenult_ver
        print(antepenult_ver)
        
        previous_ver = name.split("@")[0] + "@" + str(int(name.split("@")[1])-1)
        previous_ver = self.hidden_folder + "/"+ previous_ver
        print(previous_ver)
        
        file_name = self.hidden_folder + "/"+self.file_With_Extension(url, name)
        print(file_name)

        if path.exists(self.hidden_folder + "/"+name):
            pass

        else:
            
            if path.exists(antepenult_ver):
                shutil.rmtree(antepenult_ver)
            
            extract_path = self.hidden_folder + "/" + name
                
            urllib.request.urlretrieve(url, file_name)
            file = tarfile.open(file_name)
            file.extractall(path=extract_path)
            file.close()
            os.remove(file_name)
            
            hdf5directory = extract_path.split("/")[0] + "/" + "HDF5" + extract_path.split("/")[1]
            if(path.exists(hdf5directory)):
                pass
            else:
                os.mkdir(hdf5directory)
            
            self.make_HDF5_files(extract_path, "", hdf5directory)
            self.list_all_files(extract_path, "", previous_ver)
                

    def is_same(self, path1, path2):
        """
        Checks if two files are exactly the same or not.
        
        Parameters
        ----------
        path1: String
            path where the first file is located
        path2: String
            path where the second file is located
        
        Returns
        ----------
        Boolean Value
            Whether the given files are same or not
        """
        file1 = open(path1 , 'r') 
        data1 = file1.readlines()
        file2 = open(path2 , 'r') 
        data2 = file2.readlines()

        for i, j in zip(data1, data2):
            print("i",i)
            print("j",j)
            if i.replace(" ","")!=j.replace(" ","") :
                return False
        
        return True

    
    def extract_tabular_data(self, path):
        """
        Reads and process the file. It stores the output information in a HDF5
        file format.
        
        Parameters
        ----------
        path: String
            path where the file is located
            
        Returns
        ----------
        dataframe_dict: Dictionary
            dataframe as value and number as key
        
        metadata_dict: Dictionary
            MetaData as value and dataframe number as key
        """
        if "osc" in path.lower():
            return OSC_Parser(path)
        
        if "col" in path.lower():
            return COL_Parser(path)
        
        if "osc" in path.lower():
            return PHOT_Parser(path)
        
        file = open(path , 'r') 
        data = file.readlines()
        start, end = 0,0
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
        
        COLUMNS_ENERGY_LEVELS = ['Configuration', 'g', 'E(cm^-1)', '10^15 Hz', 'eV', 'Lam(A)', 'ID', 'ARAD', 'C4', 'C6']
        COLUMNS_OSCILLATOR_STRENGTHS = ['State A', 'State B', 'f', 'A', 'Lam(A)', 'i','j']
        
        META_PATTERN = r"^([\w\-\.]+)\s+!([\w ]+)$"
        Dataframe_PATTERN = r"[a-zA-Z0-9_\/\[\-\+.\)\()]+]*"
        
        dataframe_dict = {}
        metadata_dict = {}
        
        curr_pos = 0
        curr_metadata = []
        curr_data = []
        
        for ind,val in enumerate(data):
            
            if ind < t[curr_pos][0] :
                if re.findall( META_PATTERN, val):
                    curr_metadata.append(val)
            
            elif ind == t[curr_pos][0]:
                metadata_dict[curr_pos] = curr_metadata
                curr_data = []
            
            elif ind <= t[curr_pos][1]:
                x = re.findall(Dataframe_PATTERN, val)
                y=[]
                for j in x:
                    y.append(re.sub("^-|-$", "" , j))
                curr_data.append(y)
                
            else:
                dataframe_dict[curr_pos] = pd.DataFrame(columns = COLUMNS_ENERGY_LEVELS, data = curr_data)
                curr_metadata = []
                curr_pos += 1
                
        dataframe_dict[curr_pos] = pd.DataFrame(columns = COLUMNS_OSCILLATOR_STRENGTHS, data = curr_data)   
        
        return dataframe_dict, metadata_dict
    
    
    def write_to_HDF5(self, dataframes, MetaData, path):
        """
        Writes the given dataframes along with their metadata into a HDF5 file
        at a particular specified path
        
        Parameters
        ----------
        dataframes: Dictionary
            dataframe as value and number as key
        
        MetaData: Dictionary
            MetaData as value and dataframe number as key
            
        path: String
            path where the file is to be stored
            
        Returns
        ----------
        None
        """
        for key in dataframes:
            with pd.HDFStore(path, 'a') as f:
                f.put(str(key), dataframes[key], format='table', data_columns=True)
                f.get_storer(str(key)).attrs.metadata = MetaData[key]
    
    
    def process_file(self, path):
        """
        Reads and process the file. It stores the output information in a HDF5
        file format.
        
        Parameters
        ----------
        path: String
            path where the file is located
            
        Returns
        ----------
        None
        """
        dataframes, metadata = self.extract_tabular_data(path)
        last_path = ""
        for i in path.split("/")[2:-1]:
            last_path += "/" + i
        last_path += "/" + os.path.splitext(path.split("/")[-1])[0] +".hdf5"
        writepath = path.split("/")[0] + "/HDF5" + path.split("/")[1] + last_path
        self.write_to_HDF5( dataframes, metadata, writepath)
        
    
    def make_HDF5_files(self, basepath, directory, HDF5directory):
        """
        Makes all the HDF5 files in correct path 
        
        Parameters
        ----------
        basepath: String
            The base folder
        directory: String
            The folder we are currently on
            
        HDF5directory: String
            The folder in which HDF5 files are to be created
        Returns
        ----------
        None
        """
        extensions = [".dat", ".sp", ".txt", ""]
        
        with os.scandir(basepath) as entries:
            for entry in entries:
                
                if entry.is_file():
                    new_path = HDF5directory +"/"+ os.path.splitext(entry.name)[0] +".hdf5"
                    old_path = basepath + "/" + entry.name
                    if os.path.splitext(entry.name)[1] in extensions :
                        if os.path.exists(old_path):
                            f = open(new_path,"w")
                            f.close()
                    
                elif entry.is_dir():
                    if(path.exists(os.path.join(HDF5directory,entry.name))):
                        pass
                    else:
                        os.mkdir(os.path.join(HDF5directory,entry.name))
                    
                    self.make_HDF5_files(os.path.join(basepath,entry.name),
                                        os.path.join(directory,entry.name),
                                        os.path.join(HDF5directory,entry.name))
    
    
    def list_all_files(self, basepath, directory, prev_ver):
        """
        Finds all the files 
        
        Parameters
        ----------
        basepath: String
            The base folder of current version
        directory: String
            The folder we are currently on
        prev_ver: String
            The base folder of previous version
            
        Returns
        ----------
        None
        """
        
        extensions = [".dat", ".sp", ".txt", ""]
        with os.scandir(basepath) as entries:
            for entry in entries:
                
                if entry.is_file():
                   
                    prev_path = os.path.join(prev_ver , entry.name)
                    new_path = os.path.join(basepath , entry.name)
                    
                    if os.path.exists(prev_path):
                        if(os.path.splitext(prev_path)[1] in extensions):    
                            if self.is_same(prev_path, new_path):
                                copy_path = new_path.split("/")[0] + "/HDF5" + new_path.split("/")[1] + "/" + new_path.split("/")[2:]
                                shutil.copyfile( new_path, copy_path)
                            else:
                                self.process_file(new_path)
                        else:
                            self.process_file(new_path)
                
                elif entry.is_dir():
                    self.list_all_files(os.path.join(basepath,entry.name),
                                        os.path.join(directory,entry.name), 
                                        os.path.join(prev_ver,entry.name))
                    
                    
    def update(self, url = "http://kookaburra.phyast.pitt.edu/hillier/web/CMFGEN.htm"):
        """
        The driver function which will ultimately update the data of HDF5 files
        
        Parameters
        ----------
        url (Optional): String
            url where the links are present
        Returns
        ----------
        None
        
        """

        self.get_links(url)
        file = open(self.hidden_folder + "/CMFGEN.txt","r")
        data = file.readlines()
        t=0
        
        for i in range(len(data)):
            
            if data[i]!="\n":
                t = t + 1

            if t == 4 :
                t = 0
                self.download_data ( data[i-2][:-1], 
                                    data[i-3].split(":")[1][:-1].replace(" ","")
                                    + "@" + data[i].split(":")[1][:-1] )

