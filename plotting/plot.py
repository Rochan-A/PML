import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

#NEEDS EDITS
base_experiment_folder = ""
output_folder = "../saves/plots/"
file_ext = '.pkl'

def read_data(experiment_name):
    file = os.path.join(base_experiment_folder, experiment_name + file_ext)

    #read pickle file
    with open(file, 'rb') as f:
        data_dict = pkl.load()
    
    return data_dict

"""
def plot(data_dict, path, x_label):
    plt.figure(dpi=200)    
    
    #plt.yticks(np.arange(0, 200, 20))
    #plt.ylabel(ylabel)
    #plt.plot(data, "bs-")
    for key, value in data_dict.items():
        plt.plot(value, label=key)
    
    plt.xlabel(x_label)
    plt.legend(loc='upper right')
    #plt.title(experiment_name)

    plt.savefig(path)
"""

def plot(data_dict, output_file, title, x_label):
  
  fig, ax = plt.subplots(dpi=200)
  m = max(i for v in data_dict.values() for i in v)

  #NEEDS EDITS
  #ax.set_yticks(np.arange(0, 20, m+1))

  for key, value in data_dict.items():
        ax.plot(value, label=key)
  ax.set_xlabel(x_label)
  ax.set_title(title)
  ax.legend(loc='upper right')
  plt.savefig(output_file)

def plot_main(experiment_name, plot_aspects, title):
    
    #select aspects to plot, e.g. loss with context encoder
    full_data_dict = read_data(experiment_name)
    select_data_dict = {k:full_data_dict[k] for k in plot_aspects}

    #create dir for output plots, if not already exists
    exp_dir = os.path.join(output_folder, experiment_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    out_file = os.path.join(exp_dir, title+'.png')

    #plot 
    plot(select_data_dict, out_file, title, x_label="Trials")

