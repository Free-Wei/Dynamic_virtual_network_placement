#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 00:39:37 2024

@author: wei
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from effVFP import VNE
from warnings import warn
if __name__ == '__main__':
  warn('Warning !', DeprecationWarning)

print('========================= running =========================')
T = 500000
def filtering(y,win):
    z = np.convolve(y,np.ones(win))/win
    return z[range(win,len(y))]

def simulINS(T,M,N,k,test_times,folder_path):  
    result_test_total_acc1 = np.zeros((T, test_times))
    con_test_total_acc1 = np.zeros((T, test_times))
    running_time_acc1 = np.zeros(test_times)
    res_opt_acc1_total = np.zeros(test_times)
    for m in tqdm (range (test_times), 
               desc="Loading…", 
               ascii=False, ncols=75):
        total_power = np.random.uniform(0.5,1,M*N).reshape(M,N)
        capacity = np.random.uniform(0,1,M*k*N).reshape(M,N,k)
        lam = np.random.uniform(0.5,1,M)
        lam = lam/np.sum(lam)
        beta = np.ones((N,k))*0.1
        test_acc1 = VNE(total_power, N, M, T, lam, capacity, beta)

        start_time = time.time() 
        res_total_acc1, res_opt_acc1, con_total_acc1 = test_acc1.run_acc1()
        end_time = time.time() 
        running_time_acc1[m] = end_time - start_time
        
        res_opt_acc1_total[m] = res_opt_acc1
        result_test_total_acc1[:,m] = res_total_acc1
        con_test_total_acc1[:,m] = con_total_acc1
        

    # +++++
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
    
    return result_test_total_acc1, res_opt_acc1_total, con_test_total_acc1, running_time_acc1
    
    
# running
folder_path = "dataOUTPUT"
res_total_acc1_N10M3, _, con_test_total_N10M3,_ = simulINS(T,3,10,k,test_times,folder_path)
res_total_acc1_N20M5, _, con_test_total_N20M5,_ = simulINS(T,5,20,k,test_times,folder_path)
res_total_acc1_N50M8, _, con_test_total_N50M8,_ = simulINS(T,8,50,k,test_times,folder_path)
res_total_acc1_N100M10, _, con_test_total_N100M10,_ = simulINS(T,10,100,k,test_times,folder_path)


# +++++++++++++++++++ Plot +++++++++++++++++++

def plot_instances(res_total_acc1_N10M3, res_total_acc1_N20M5,
            res_total_acc1_N50M8, res_total_acc1_N100M10, judge):
    win = 1000
    sub_samples = 5000
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(T)

    y = filtering(res_total_acc1_N10M3.mean(axis = 1), win)
    y1 = filtering(res_total_acc1_N20M5.mean( axis = 1), win)
    y2 = filtering(res_total_acc1_N50M8.mean( axis = 1), win)
    y3 =filtering(res_total_acc1_N100M10.mean( axis = 1), win)

    ax.plot(x[win::sub_samples], y[::sub_samples], lw=2, label='N=10,M=3', color='blue')
    ax.fill_between(x[win::sub_samples], 
                    filtering(np.max(res_total_acc1_N10M3, axis = 1), win)[::sub_samples],
                    filtering(np.min(res_total_acc1_N10M3, axis = 1), win)[::sub_samples], facecolor='blue', alpha=0.2)
    
    ax.plot(x[win::sub_samples], y1[::sub_samples], lw=2, label='N=20,M=5', color='red')
    ax.fill_between(x[win::sub_samples], 
                    filtering(np.max(res_total_acc1_N20M5, axis = 1), win)[::sub_samples],
                    filtering(np.min(res_total_acc1_N20M5, axis = 1), win)[::sub_samples], facecolor='red', alpha=0.2)
    
    ax.plot(x[win::sub_samples], y2[::sub_samples], lw=2, label='N=50,M=8', color='green')
    ax.fill_between(x[win::sub_samples],
                    filtering(np.max(res_total_acc1_N50M8, axis = 1), win)[::sub_samples],
                    filtering(np.max(res_total_acc1_N50M8, axis = 1), win)[::sub_samples], facecolor='green', alpha=0.2)
    
    ax.plot(x[win::sub_samples], y3[::sub_samples], lw=2, label='N=100,M=10', color='black')
    ax.fill_between(x[win::sub_samples], 
                    filtering(np.max(res_total_acc1_N100M10, axis = 1), win)[::sub_samples],
                    filtering(np.max(res_total_acc1_N100M10, axis = 1), win)[::sub_samples], facecolor='black', alpha=0.1)
    if judge == 1:
        ax.plot(x, np.array(1).repeat(T), lw=3, linestyle='dashed', color='black')
        ax.set_ylabel('Maximum Constraint Vaule')
        ax.legend(fontsize = 20,loc='center right')   
    else:
        ax.set_ylabel('Relative Gap')
        ax.legend(fontsize = 20,loc='upper right')   
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    ax.set_xlabel('Iterations')
    ax.set_xlim([0,T])
    ax.grid()
    plt.savefig(name, format="pdf", bbox_inches="tight")
plot_instance(res_total_acc1_N10M3, res_total_acc1_N20M5, res_total_acc1_N50M8, res_total_acc1_N100M10, f'img/output_T{T}_{test_times}run_N{N}_M{M}_acc1.pdf')
plot_instance(con_test_total_N10M3, con_test_total_N20M5, con_test_total_N50M8, con_test_total_N100M10, f'img/output_T{T}_{test_times}run_N{N}_M{M}_acc1_constraints.pdf', 1)
        
print(f'========================= Data saved to file {folder_path} and image saved to file img =========================')
