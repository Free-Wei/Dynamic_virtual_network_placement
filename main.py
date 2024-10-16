#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:33:50 2024

@author: wei
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.effVFP import VNE
from plot_diff_algo import *
from plot_diff_instances import *

import warnings
warnings.filterwarnings("ignore")




print('========================= running 1 =========================')
# ============= 
# Parameter 
# ============= 
M = 3
N = 10
k = 2
T = N*M*10
test_times = 50
    
# running
folder_path = "dataOUTPUT"
result_test_total, con_test_total, running_time, result_test_total_acc1, con_test_total_acc1, running_time_acc1 = simulALGO(T,M,N,k,test_times,folder_path)


plot_algos(result_test_total, result_test_total_acc1, f'img/output_T{T}_{test_times}run_N{N}_M{M}_acc1.pdf')
plot_algos(con_test_total, con_test_total_acc1, f'img/output_T{T}_{test_times}run_N{N}_M{M}_acc1_constraints.pdf', 1)
        
print(f'========================= Data saved to file {folder_path} and image saved to file img =========================')



print('========================= running 2 =========================')
T = 500000
    
    
# running
folder_path = "dataOUTPUT"
res_total_acc1_N10M3, _, con_test_total_N10M3,_ = simulINS(T,3,10,k,test_times,folder_path)
res_total_acc1_N20M5, _, con_test_total_N20M5,_ = simulINS(T,5,20,k,test_times,folder_path)
res_total_acc1_N50M8, _, con_test_total_N50M8,_ = simulINS(T,8,50,k,test_times,folder_path)
res_total_acc1_N100M10, _, con_test_total_N100M10,_ = simulINS(T,10,100,k,test_times,folder_path)


# +++++++++++++++++++ Plot +++++++++++++++++++
plot_instances(res_total_acc1_N10M3, res_total_acc1_N20M5, res_total_acc1_N50M8, res_total_acc1_N100M10, f'img/output_T{T}_{test_times}run_N{N}_M{M}_acc1.pdf')
plot_instances(con_test_total_N10M3, con_test_total_N20M5, con_test_total_N50M8, con_test_total_N100M10, f'img/output_T{T}_{test_times}run_N{N}_M{M}_acc1_constraints.pdf', 1)
        
print(f'========================= Data saved to file {folder_path} and image saved to file img =========================')
