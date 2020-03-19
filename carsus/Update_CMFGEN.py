#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 21:40:38 2020

@author: piyush
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

class UPDATE_CMFGEN():
    
    """     
    Class Members
        ----------
        hidden_folder : String
            It is the name of the folder which is hidden and where all the data 
            is stored. The data includes the extracted file, log file and HDF5 
            files. 
            
    Member Functions
        ----------
        get_links : 
         
    """
    
    hidden_folder = ".CMFGEN"
    
    
    def __init__(self, *args, **kwargs):
        if args:    
            if(args[0][0]=='.' and args[0]!='.'):
                self.hidden_folder = args[0] 
            else:
                print("The given folder was not hidden.")
                print("The default folder will be used.")
                
        
    def get_links(self):
        """
        Fetches all links and also update the log Table.
        
        Parameters
        ----------
        None

        Returns
        ----------
        None
        """

        url="http://kookaburra.phyast.pitt.edu/hillier/web/CMFGEN.htm"
        html_content = requests.get(url).text
        first = "http://kookaburra.phyast.pitt.edu/hillier/"
        
        soup = BeautifulSoup(html_content, "lxml")
        file = open( self.hidden_folder + "/CMFGEN.txt" , "r")
        file_new = open( self.hidden_folder + "/CMFGEN2.txt" ,"w")
        
        for link, date in zip (soup.find_all("a")[2:-1], soup.find_all("dd")[:-1]):
            
            new_text = link.text
            new_href = link.get("href")
            new_href = first + new_href.split("../")[1]
            new_date = date.text[1:]
            
            prev_text = file.readline()
            prev_href = file.readline()
            prev_date = file.readline()
            prev_version = file.readline()
            file.readline()
            
            if prev_date != new_date+"\n":
                file_new.write("Inner Text: {}".format(new_text)+"\n")
                file_new.write(new_href)
                file_new.write(date.text+"\n")
                print(str(int(prev_version.split(":")[1])+1))
                file_new.write("Version:"+str(int(prev_version.split(":")[1])+1))
                file_new.write("\n")
            
            else:
                file_new.write(prev_text)
                file_new.write(prev_href)
                file_new.write(prev_date)
                file_new.write(prev_version)
                file_new.write("\n")

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
        elif url.endswith("tar.gz"):
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
            
            with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
                extract_path = file_name.split(".")[1]
                extract_path = "."+extract_path
                
                if file_name.endswith("tar.gz"):
                    tar = tarfile.open(file_name, "r:gz")
                    tar.extractall(extract_path+"/")
                    tar.close()
                    os.remove(file_name)
                
                elif file_name.endswith("tar"):
                    tar = tarfile.open(file_name, "r:")
                    tar.extractall(extract_path+"/")
                    tar.close()
                    os.remove(file_name)
            print("extract_path", extract_path)
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
        """
        
        file1 = open(path1 , 'r') 
        data1 = file1.readlines()
        file2 = open(path2 , 'r') 
        data2 = file2.readlines()
        
        for i, j in zip(data1, data2):
            
            if i!=j :
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
        
        META_PATTERN = "^([\w\-\.]+)\s+!([\w ]+)$"
        Dataframe_PATTERN = "[a-zA-Z0-9_\/\[\-\+.\)\()]+]*"
        
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
        hf = h5py.File(path, 'w')
        hf.close()
        store = pd.HDFStore(path)
        
        for key in dataframes: 
            store.put( str(key) , dataframes[key])
            store.get_storer(str(key)).attrs.metadata = MetaData
            
        store.close()
    
    
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
        print(path.split("/"))
        last_path = ""
        for i in path.split("/")[2:-1]:
            last_path += "/" + i
        last_path += "/" + os.path.splitext(path.split("/")[-1])[0] +".h5"
        writepath = path.split("/")[0] + "/HDF5" + path.split("/")[1] + last_path
        print(writepath)
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
        with os.scandir(basepath) as entries:
            for entry in entries:
                
                if entry.is_file():
                    new_path = HDF5directory +"/"+ os.path.splitext(entry.name)[0] +".h5"
                    f = open(new_path,"w")
                    f.close()
                    
                elif entry.is_dir():
                    if(path.exists(os.path.join(HDF5directory,entry.name))):
                        pass
                    else:
                        print(os.path.join(HDF5directory,entry.name))
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
        hdf5directory = basepath.split("/")[0] + "/" + "HFD5" + basepath.split("/")[1]
        if(path.exists(hdf5directory)):
            pass
        else:
            os.mkdir(hdf5directory)
        
        self.make_HDF5_files(basepath, directory, hdf5directory)
        
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
                    
                    
    def update(self):
        """
        The driver function which will ultimately update the data of HDF5 files
        
        Parameters
        ----------
        None

        Returns
        ----------
        None
        """

        self.get_links()
        file = open(self.hidden_folder + "/CMFGEN.txt","r")
        data = file.readlines()
        t=0

        for i in range(len(data)):
            
            if data[i]!="\n" :
                t = t + 1

            if t == 4 :
                t = 0
                self.download_data ( data[i-2][:-1], 
                                    data[i-3].split(":")[1][:-1].replace(" ","")
                                    + "@" + data[i].split(":")[1][:-1] )

a = UPDATE_CMFGEN()