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

from effVFP import VNE
import warnings
warnings.filterwarnings('ignore')

print('========================= running =========================')
# ============= 
# Parameter 
# ============= 
M = 3
N = 10
k = 2

T = N*M*1000
np.random.seed(30)
total_power = np.random.uniform(0.5,1,M*N).reshape(M,N)
capacity = np.random.uniform(0,1,M*k*N).reshape(M,N,k)
lam = np.random.uniform(0.5,1,M)
lam = lam/np.sum(lam)

beta = np.ones((N,k))*0.1
test_times = 1

def filtering(y,win):
    z = np.convolve(y,np.ones(win))/win
    return z[range(win,len(y))]

def simulINS_1(T,M,N,k,test_times, folder_path):  
    result_test_total = np.zeros((T, test_times))
    con_test_total = np.zeros((T, test_times))
    running_time = np.zeros(test_times)
    res_opt_total = np.zeros(test_times)

    result_test_total_acc1 = np.zeros((T, test_times))
    con_test_total_acc1 = np.zeros((T, test_times))
    running_time_acc1 = np.zeros(test_times)
    res_opt_acc1_total = np.zeros(test_times)

    for m in range(test_times):

        test = VNE(total_power, N, M, T, lam, capacity, beta)

        start_time = time.time() 
        res_total, res_opt, con_total = test.run()
        end_time = time.time() 
        running_time[m] = end_time - start_time

        result_test_total[:,m] = res_total
        con_test_total[:,m] = con_total


        test_acc1 = VNE(total_power, N, M, T, lam, capacity, beta)

        start_time = time.time() 
        res_total_acc1, res_opt_acc1, con_total_acc1 = test_acc1.run_acc1()
        end_time = time.time() 
        running_time_acc1[m] = end_time - start_time

        res_opt_acc1_total[m] = res_opt_acc1
        result_test_total_acc1[:,m] = res_total_acc1
        con_test_total_acc1[:,m] = con_total_acc1

    # +++++
    data_default = {
    'max_result': np.max(result_test_total, axis = 1),
    'min_result': np.min(result_test_total, axis = 1),
    'mean_result': result_test_total.mean( axis = 1), 
    'max_con': np.max(con_test_total, axis = 1), 
    'min_con': np.min(con_test_total, axis = 1),
    'mean_con': con_test_total.mean(axis =1),
    'mean_running_time': np.mean(running_time),
    'max_running_time': np.max(running_time), 
    'min_running_time': np.min(running_time), 
    }
    df = pd.DataFrame(data_default)
    df.to_csv(f'{folder_path}/changed_output_T{T}_{test_times}run_N{N}_M{M}_default.csv', index=False)
    
    data_acc1 = {
    'max_result': np.max(result_test_total_acc1, axis = 1),
    'min_result': np.min(result_test_total_acc1, axis = 1),
    'mean_result': result_test_total_acc1.mean( axis = 1),
    'max_con': np.max(con_test_total_acc1, axis = 1),
    'min_con': np.min(con_test_total_acc1, axis = 1),
    'mean_con': con_test_total_acc1.mean(axis =1),
    'mean_running_time': np.mean(running_time_acc1),
    'max_running_time': np.max(running_time_acc1), 
    'min_running_time': np.min(running_time_acc1), 
    'max_opt':np.max(res_opt_acc1_total),
    'min_opt':np.min(res_opt_acc1_total), 
    'mean_opt':np.mean(res_opt_acc1_total),
    }

    df_acc1 = pd.DataFrame(data_acc1)
    df_acc1.to_csv(f'{folder_path}/changed_output_T{T}_{test_times}run_N{N}_M{M}_acc1.csv', index=False)
    
    return result_test_total, con_test_total, running_time, result_test_total_acc1, con_test_total_acc1, running_time_acc1
    
    
# running
folder_path = "dataOUTPUT"
result_test_total, con_test_total, running_time, result_test_total_acc1, con_test_total_acc1, running_time_acc1 = simulINS_1(T,M,N,k,test_times,folder_path)


# +++++++++++++++++++ Plot +++++++++++++++++++
plotSIM1(result_test_total, result_test_total_acc1, "img/test_T500000_algos_sub_final.pdf")
plotSIM1(con_test_total, con_test_total_acc1, "img/test_T500000_algos_sub_constraint_final.pdf", 1)
        
print('========================= file saved =========================')

