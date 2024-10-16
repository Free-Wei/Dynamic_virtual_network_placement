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
from tqdm import tqdm
from src.effVFP import VNE
import warnings
warnings.filterwarnings("ignore")


print('========================= running =========================')
M = 3
N = 10
k = 2

T = N*M*1000
np.random.seed(30)
c = np.random.uniform(0.5,1,M*N).reshape(M,N)
a = np.random.uniform(0,1,M*k*N).reshape(M,N,k)
lam = np.random.uniform(0.5,1,M)
lam = lam/np.sum(lam)

beta = np.ones((N,k))*0.1
test_times = 50
def filtering(y,win):
    z = np.convolve(y,np.ones(win))/win
    return z[range(win,len(y))]

def simulALGO(T,M,N,k,test_times, folder_path):  
    result_test_total = np.zeros((T, test_times))
    con_test_total = np.zeros((T, test_times))
    running_time = np.zeros(test_times)
    res_opt_total = np.zeros(test_times)

    result_test_total_acc1 = np.zeros((T, test_times))
    con_test_total_acc1 = np.zeros((T, test_times))
    running_time_acc1 = np.zeros(test_times)
    res_opt_acc1_total = np.zeros(test_times)

    for m in tqdm (range (test_times), 
               desc="Loading…", 
               ascii=False, ncols=75):

        test = VNE(c, N, M, T, lam, a, beta)

        start_time = time.time() 
        res_total, res_opt, con_total = test.run()
        end_time = time.time() 
        running_time[m] = end_time - start_time

        result_test_total[:,m] = res_total
        con_test_total[:,m] = con_total


        test_acc1 = VNE(c, N, M, T, lam, a, beta)

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
result_test_total, con_test_total, running_time, result_test_total_acc1, con_test_total_acc1, running_time_acc1 = simulALGO(T,M,N,k,test_times,folder_path)


# +++++++++++++++++++ Plot +++++++++++++++++++
def plot_algos(result_test_total, result_test_total_acc1, name, judge=0):
    win = 500
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(T)
    r = range(win,T)

    y = filtering(result_test_total.mean( axis = 1), win)
    y1 = filtering(result_test_total_acc1.mean( axis = 1), win)

    ax.plot(x[r], y, lw=2, label='Base algorithm', color='blue')
    ax.fill_between(x[r], 
                    filtering(np.max(result_test_total, axis = 1), win), 
                    filtering(np.min(result_test_total, axis = 1), win), facecolor='blue', alpha=0.2)
    ax.plot(x[r], y1, lw=2, label='Fast algorithm', color='red')
    ax.fill_between(x[r], 
                    filtering(np.max(result_test_total_acc1, axis = 1), win), 
                    filtering(np.min(result_test_total_acc1, axis = 1), win), facecolor='red', alpha=0.2)
    if judge == 1:
        ax.plot(x, np.array(1).repeat(T),linestyle='dashed', lw=3, color='black')
        plt.text(12500, 1.03, 'Threshold', fontsize = 20)
        ax.set_ylabel('Maximum Constraint Vaule')
        ax.legend(fontsize = 25,loc='lower right')
    else:
        ax.set_ylabel('Relative Gap')
        ax.legend(fontsize = 25,loc='upper right')
    ax.set_xlabel('Iterations')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
		ax.get_xticklabels() + ax.get_yticklabels()):
    	item.set_fontsize(20)
    ax.set_xlim([0,T])
    ax.grid()
    plt.savefig(name, format="pdf", bbox_inches="tight")
plot_algos(result_test_total, result_test_total_acc1, f'img/output_T{T}_{test_times}run_N{N}_M{M}_acc1.pdf')
plot_algos(con_test_total, con_test_total_acc1, f'img/output_T{T}_{test_times}run_N{N}_M{M}_acc1_constraints.pdf', 1)
        
print(f'========================= Data saved to file {folder_path} and image saved to file img =========================')

