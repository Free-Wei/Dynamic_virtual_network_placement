#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:33:50 2024

@author: wei
"""
import numpy as np
import matplotlib.pyplot as plt

import collections
import os
import copy
import logging
import pickle
import time
from tqdm import tqdm
import math
import itertools
import scipy.optimize as optimize
import pandas as pd

from warnings import warn
if __name__ == '__main__':
  warn('Warning !', DeprecationWarning)


class VNE:
    def __init__(self, c, N, M, T, lam, a, beta):
        """Class constructor ...
        See paper
        """
        self.T = T
        
        self.M = M
        
        self.N = N

        self.t = 0
        
        self.rho = 1.05
        
        self.a = a
        
        self.c = c
        
        self.lam = lam

        self.beta = beta

        self.pulls = np.zeros((self.M, self.N),  dtype=np.int32)
        
        self.c_cumu = np.zeros((self.M, self.N))
        
        self.a_cumu = np.zeros_like(self.a)
        
        self.a_con_cumu= np.zeros_like(self.a)
                
        self.lam_cumu = np.zeros(self.N)
        
        self.c_hat = np.zeros((self.M, self.N))
        
        self.a_hat = np.zeros_like(self.a)
        
        self.lam_hat = np.zeros(self.M)
        
        self.update_times = np.unique([self.N*self.M+np.round(np.power( self.rho,k)) for k in np.arange(np.ceil(np.log(self.T)/np.log(self.rho)))])
        
        self.kl_a = np.zeros_like(self.a)
        
        self.kl_c = np.zeros((self.M, self.N))
        
        self.kl_lam_up = np.zeros(self.M)
        
        self.kl_lam_dn = np.zeros(self.M)
        
        self.beta_new = np.tile(self.beta,self.M).reshape((self.N,np.shape(a)[-1],self.M))
    #Compute KL divergence
    def KL(self, p, q):
        if p == 0:
            return math.log(1 / (1 - q))
        if p == 1:
            if q == 0:
                return 1e6
            return math.log(1 / q)
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
    
    # First derivative of KL divergence w.r.t q
    def kl_prime(self, p, q):
        return -p / q + (1 - p) / (1 - q)
    
    # KL-Upper Confidence Bound
    def getUCBKL(self, estimated, sample_times,lb,ub,tol = 1e-7):
        if sample_times == 0:
            kl = ub
        else:
            bound = (math.log(self.t)) / sample_times 
            if estimated == ub:
                kl = ub
            else:
                q = (ub+estimated)/2
                while self.KL(estimated, q) < bound: 
                    q= (q+ub)/2
                compute_kl = self.KL(estimated, q)
                while np.abs(compute_kl -  bound) > tol:
                    compute_kl = self.KL(estimated, q)
                    compute_kl_prime = self.kl_prime(estimated, q)
                    q -= (compute_kl-bound)/compute_kl_prime
                kl = q
        return kl
    # KL-Lower Confidence Bound    
    def getLCBKL(self, estimated, sample_times,lb,ub ,tol = 1e-7):
        if sample_times == 0:
            kl = lb
        else:
            bound = (math.log(self.t)) / sample_times 
            if estimated == 0:
                kl = lb
            else:
                q = (lb+estimated)/2
                while self.KL(estimated, q) < bound: 
                    q = (q+lb)/2
                compute_kl = self.KL(estimated, q)
                while np.abs(compute_kl -  bound) > tol:
                    compute_kl = self.KL(estimated, q)
                    compute_kl_prime = self.kl_prime(estimated, q)
                    q -= (compute_kl-bound)/compute_kl_prime
                kl = q
        return kl
    
    def compute_power(self, stra, opt_stra): 
        return np.sum(np.ravel(self.c.T*np.tile(self.lam,self.N).reshape((self.N, self.M))) * np.ravel((stra-opt_stra).T)) / np.sum(np.ravel(self.c.T*np.tile(self.lam,self.N).reshape((self.N, self.M))) * np.ravel(opt_stra.T))


    
    def compute_utilization(self, arms_t):
        utilization = np.zeros((self.N, np.shape(self.a)[-1]))
        for arms in arms_t:
            for k in range(np.shape(self.a)[-1]):
                utilization[arms, k] = np.sum(self.a_con_cumu[:, arms, k])/(self.beta[arms, k]*self.t)
        return np.max(utilization)
    
    def enter(self):
        lam = np.zeros(self.M)
        player = np.random.choice(self.M, 1, p=self.lam)
        lam[player] = 1
        return lam.astype(int)
    
    def draw(self, arms, lam):
        c = np.zeros((self.M, self.N))
        a = np.zeros_like(self.a)
        for player in range(self.M):
            if lam[player] != 0:
                c[player][arms[player]] = np.random.binomial(1,self.c[player][arms[player]])
                for a in range(self.N):
                    for k in range(np.shape(self.a)[-1]):
                        a[player,a,k] = np.random.binomial(1,self.a[player,a,k])
        return c, a

    def lp(self, c, a, lam_dn):
        obj_mul = np.ravel(c.T*np.tile(lam_dn,self.N).reshape((self.N, self.M)))
        a = np.transpose(a, (1, 2, 0))/self.beta_new
        A_ub = np.zeros((self.N*np.shape(a)[-1],len(obj_mul)))
        count = 0
        j = 0
        for i in range(self.N):
            for m in range(np.shape(a)[-1]):
                for k in range(self.M):
                    A_ub[j,count+k] = a[i,m,k]
                j += 1
            count += self.M
        B_ub = np.ones(self.N*np.shape(a)[-1])
        A_eq = np.tile(np.eye(self.M),self.N)
        B_eq = np.ones(self.M)
        LP_solution = optimize.linprog( obj_mul.T, A_ub, B_ub.T, A_eq, B_eq.T)
        return LP_solution.x, LP_solution.fun
    
    def lp_acc3(self, stra_init, c, a, lam_dn):
        obj_mul = np.ravel(c.T*np.tile(lam_dn,self.N).reshape((self.N, self.M)))
        a = np.transpose(a, (1, 2, 0))/self.beta_new
        A_ub = np.zeros((self.N*np.shape(a)[-1],len(obj_mul)))
        count = 0
        j = 0
        for i in range(self.N):
            for m in range(np.shape(a)[-1]):
                for k in range(self.M):
                    A_ub[j,count+k] = a[i,m,k]
                j += 1
            count += self.M
        B_ub = np.ones(self.N*np.shape(a)[-1])
        A_eq = np.tile(np.eye(self.M),self.N)
        B_eq = np.ones(self.M)
        LP_solution = optimize.linprog( obj_mul.T, A_ub, B_ub.T, A_eq, B_eq.T, x0 = stra_init)
        return LP_solution.x, LP_solution.fun
    
    def lp_opt(self):
        obj_mul = np.ravel(self.c.T*np.tile(self.lam,self.N).reshape((self.N, self.M)))
        a = np.transpose(self.a, (1, 2, 0))/self.beta_new*np.tile(np.tile(self.lam,self.N).reshape((self.N, self.M)),np.shape(self.a)[-1]).reshape(self.N,np.shape(self.a)[-1],self.M)
        A_ub = np.zeros((self.N*np.shape(self.a)[-1],len(obj_mul)))
        count = 0
        j = 0
        for i in range(self.N):
            for m in range(np.shape(self.a)[-1]):
                for k in range(self.M):
                    A_ub[j,count+k] = a[i,m,k]
                j += 1
            count += self.M
        B_ub = np.ones(self.N*np.shape(self.a)[-1])
        A_eq = np.tile(np.eye(self.M),self.N)
        B_eq = np.ones(self.M)
        LP_solution = optimize.linprog( obj_mul.T, A_ub, B_ub.T, A_eq, B_eq.T)
        #LP_solution = optimize.linprog( obj_mul.T, A_ub, B_ub.T, A_eq, B_eq.T,options={'tol': 1e-12})
        opt_allo = LP_solution.x.reshape(self.N,self.M)
        strategy = np.zeros((self.M,self.N))
        nor_opt = np.zeros((self.M,self.N))
        for player in range(self.M):
            strategy[player,:] = opt_allo[:,player]
            nor_opt[player,:] = strategy[player,:]/np.sum(strategy[player,:])
        #print("strategy", strategy)
        opt = np.sum(np.ravel(self.c.T*np.tile(self.lam,self.N).reshape((self.N, self.M))) * np.ravel(nor_opt.T))
        print("opt", opt)
        return LP_solution.x, opt
    
    def allocation(self, allo):
        arm_chosen = np.zeros(self.M)
        allocation = allo.reshape(self.N,self.M)
        strategy = np.zeros((self.M,self.N))
        nor_strategy = np.zeros((self.M,self.N))
        for player in range(self.M):
            strategy[player,:] = allocation[:,player]
            nor_strategy[player,:] = strategy[player,:]/np.sum(strategy[player,:])
            arm_chosen[player] = np.random.choice(self.N, 1, p = nor_strategy[player,:])
        return arm_chosen, np.ravel(nor_strategy.T)
        
    def allocation_acc1(self, allo):
        arm_chosen = np.zeros(self.M)
        allocation = allo.reshape(self.N,self.M)
        strategy = np.zeros((self.M,self.N))
        nor_strategy = np.zeros((self.M,self.N))
        for player in range(self.M):
            strategy[player,:] = allocation[:,player]
            for a in range(self.N):
                if strategy[player,a] < 0.001:
                    strategy[player,a]  = 0.01 -(self.t/self.T)*0.01
            nor_strategy[player,:] = strategy[player,:]/np.sum(strategy[player,:])
            #print(nor_strategy)
            arm_chosen[player] = np.random.choice(self.N, 1, p = nor_strategy[player,:])
        #print(nor_strategy)
        return arm_chosen, np.ravel(nor_strategy.T)
    
    def choose_arm(self):
        for player in range(self.M):
            #self.kl_lam_up[player] = self.getUCBKL(self.lam_hat[player],self.t,0,1)
            self.kl_lam_dn[player] = self.getLCBKL(self.lam_hat[player],self.t,0,1)
            for arm in range(self.N):
                self.kl_c[player, arm] = self.getLCBKL(self.c_hat[player, arm],self.pulls[player, arm],0,1)
                for G in range(np.shape(self.a)[-1]):
                    self.kl_a[player, arm,G] = self.getUCBKL(self.a_hat[player, arm, G],self.t,0,1)
        allo, _ = self.lp( self.kl_c, self.kl_a, self.kl_lam_dn )
        arm_chosen, strategy = self.allocation(allo)
        return arm_chosen.astype(int), strategy
    
    def choose_arm_acc1(self, stra_init):
        if self.t in self.update_times:
            for player in range(self.M):
                #self.kl_lam_up[player] = self.getUCBKL(self.lam_hat[player],self.t,0,1)
                self.kl_lam_dn[player] = self.getLCBKL(self.lam_hat[player],self.t,0,1)
                for arm in range(self.N):
                    self.kl_c[player, arm] = self.getLCBKL(self.c_hat[player, arm],self.pulls[player, arm],0,1)
                    for G in range(np.shape(self.a)[-1]):
                        self.kl_a[player, arm,G] = self.getUCBKL(self.a_hat[player, arm, G], self.t,0,1)
            allo, _ = self.lp(self.kl_c, self.kl_a, self.kl_lam_dn)
            arm_chosen, strategy = self.allocation(allo)
            return arm_chosen.astype(int), strategy
        elif (self.t-1 in self.update_times):
            arm_chosen, strategy = self.allocation_acc1(stra_init)
            return arm_chosen.astype(int), strategy
        else:
            arm_chosen, strategy = self.allocation(stra_init)
            return arm_chosen.astype(int), strategy            
        
    def choose_arm_acc2(self, arms_t, c = 0):
        for player in range(self.M):
                #self.kl_lam_up[player] = self.getUCBKL(self.lam_hat[player],self.t,0,1)
                self.kl_lam_dn[player] = self.getLCBKL(self.lam_hat[player],self.t,0,1)
                for arm in range(self.N):
                    if arm == arms_t[player]:
                        self.kl_c[player, arm] = self.getLCBKL(self.c_hat[player, arm],self.pulls[player, arm],0,1)
                        for G in range(np.shape(self.a)[-1]):
                            self.kl_a[player, arm,G] = self.getUCBKL(self.a_hat[player, arm, G], self.t,0,1)
        allo, _ = self.lp( self.kl_c, self.kl_a,self.kl_lam_dn )
        arm_chosen, strategy = self.allocation(allo)
        return arm_chosen.astype(int), strategy
    
    def choose_arm_acc3(self, stra_init):
        for player in range(self.M):
                #self.kl_lam_up[player] = self.getUCBKL(self.lam_hat[player],self.t,0,1)
                self.kl_lam_dn[player] = self.getLCBKL(self.lam_hat[player],self.t,0,1)
                for arm in range(self.N):
                    self.kl_c[player, arm] = self.getLCBKL(self.c_hat[player, arm], self.pulls[player, arm],0,1)
                    for G in range(np.shape(self.a)[-1]):
                        self.kl_a[player, arm,G] = self.getUCBKL(self.a_hat[player, arm, G], self.t,0,1)
        allo, _ = self.lp_acc3( stra_init, self.kl_c, self.kl_a, self.kl_lam_dn )
        arm_chosen, strategy = self.allocation(allo)
        return arm_chosen.astype(int), strategy
    
            
    def run(self):
        result_total = []
        cons_total = []
        opt_strategy, opt_result = self.lp_opt()
        while self.t < self.T: # explore
            self.t +=1
            lam_t = self.enter()
            self.lam_cumu[np.arange(self.M)] += lam_t[np.arange(self.M)]
            self.lam_hat= self.lam_cumu / (self.t)
            arms_t, strategy = self.choose_arm()
            c_t, a_t = self.draw(arms_t,lam_t)
            self.pulls[np.arange(self.M), arms_t] += lam_t
            self.c_cumu[np.arange(self.M), arms_t] += c_t[np.arange(self.M), arms_t]
            self.a_con_cumu[np.arange(self.M), arms_t,:] += a_t[np.arange(self.M), arms_t,:]
            self.a_cumu[np.arange(self.M), :,:] += a_t[np.arange(self.M), :,:]
            self.c_hat = np.divide(self.c_cumu, self.pulls,  out=np.zeros_like(self.c_cumu), where=self.pulls != 0)
            self.a_hat = np.divide(self.a_cumu, self.t, out=np.zeros_like(self.a_cumu), where=self.t != 0)
            cons = self.compute_utilization(arms_t)
            cons_total.append(cons)
            result = self.compute_power(strategy.reshape(self.N,self.M).T,(opt_strategy.reshape(self.N,self.M).T))
            result_total.append(result)
        return result_total, opt_result, cons_total
    
    
    def run_acc1(self):
        result_total = []
        cons_total = []
        opt_strategy, opt_result = self.lp_opt()
        identity_matrix = np.eye(self.N)
        stra_init = np.tile(np.ones(self.N)*(1/(self.N)),self.M)
        while self.t < self.T: # explore
            if self.t <= self.N*self.M:
                self.t += 1
                lam_t = self.enter()
                self.lam_cumu[np.arange(self.M)] += lam_t[np.arange(self.M)]
                self.lam_hat= self.lam_cumu / (self.t)
                arms_t = np.random.choice(self.N, self.M)
                c_t, a_t = self.draw(arms_t,lam_t)
                self.pulls[np.arange(self.M), arms_t] += lam_t
                self.c_cumu[np.arange(self.M), arms_t] += c_t[np.arange(self.M), arms_t]
                self.a_con_cumu[np.arange(self.M), arms_t,:] += a_t[np.arange(self.M), arms_t,:]
                self.a_cumu[np.arange(self.M), :,:] += a_t[np.arange(self.M), :,:]
                self.c_hat = np.divide(self.c_cumu, self.pulls,  out=np.zeros_like(self.c_cumu), where=self.pulls != 0)
                self.a_hat = np.divide(self.a_cumu, self.t, out=np.zeros_like(self.a_cumu), where=self.t != 0)
                cons = self.compute_utilization(arms_t)
                cons_total.append(cons)
                result = self.compute_power(stra_init.reshape(self.N,self.M).T,(opt_strategy.reshape(self.N,self.M).T))
                result_total.append(result)
            else:
                self.t +=1
                lam_t = self.enter()
                self.lam_cumu[np.arange(self.M)] += lam_t[np.arange(self.M)]
                self.lam_hat= self.lam_cumu / (self.t)
                arms_t, stra_init = self.choose_arm_acc1(stra_init)
                c_t, a_t = self.draw(arms_t,lam_t)
                self.pulls[np.arange(self.M), arms_t] += lam_t
                self.c_cumu[np.arange(self.M), arms_t] += c_t[np.arange(self.M), arms_t]
                self.a_con_cumu[np.arange(self.M), arms_t,:] += a_t[np.arange(self.M), arms_t,:]
                self.a_cumu[np.arange(self.M), :,:] += a_t[np.arange(self.M), :,:]
                self.c_hat = np.divide(self.c_cumu, self.pulls,  out=np.zeros_like(self.c_cumu), where=self.pulls != 0)
                self.a_hat = np.divide(self.a_cumu, self.t, out=np.zeros_like(self.a_cumu), where=self.t != 0)
                cons = self.compute_utilization(arms_t)
                cons_total.append(cons)
                result = self.compute_power(stra_init.reshape(self.N,self.M).T,(opt_strategy.reshape(self.N,self.M).T))
                result_total.append(result)
        return result_total, opt_result, cons_total
    
    def run_acc2(self):
        result_total = []
        cons_total = []
        opt_strategy, opt_result = self.lp_opt()
        arms_t = np.zeros(self.M)
        while self.t < self.T: # explore
            self.t +=1
            lam_t = self.enter()
            self.lam_cumu[np.arange(self.M)] += lam_t[np.arange(self.M)]
            self.lam_hat= self.lam_cumu / (self.t)
            arms_t, strategy = self.choose_arm_acc2(arms_t)
            c_t, a_t = self.draw(arms_t,lam_t)
            self.pulls[np.arange(self.M), arms_t] += lam_t
            self.c_cumu[np.arange(self.M), arms_t] += c_t[np.arange(self.M), arms_t]
            self.a_con_cumu[np.arange(self.M), arms_t,:] += a_t[np.arange(self.M), arms_t,:]
            self.a_cumu[np.arange(self.M), :,:] += a_t[np.arange(self.M), :,:]
            self.c_hat = np.divide(self.c_cumu, self.pulls,  out=np.zeros_like(self.c_cumu), where=self.pulls != 0)
            self.a_hat = np.divide(self.a_cumu, self.t, out=np.zeros_like(self.a_cumu), where=self.t != 0)
            cons = self.compute_utilization(arms_t)
            cons_total.append(cons)
            result = self.compute_power(strategy.reshape(self.N,self.M).T,(opt_strategy.reshape(self.N,self.M).T))
            result_total.append(result)
        return result_total, opt_result, cons_total
    
    def run_acc3(self):
        result_total = []
        cons_total = []
        opt_strategy, opt_result = self.lp_opt()
        identity_matrix = np.eye(self.N)
        stra_init = np.tile(identity_matrix[0,:],self.M)
        while self.t < self.T: # explore
            self.t +=1
            lam_t = self.enter()
            self.lam_cumu[np.arange(self.M)] += lam_t[np.arange(self.M)]
            self.lam_hat= self.lam_cumu / (self.t)
            arms_t, stra_init = self.choose_arm_acc3(stra_init)
            c_t, a_t= self.draw(arms_t,lam_t)
            self.pulls[np.arange(self.M), arms_t] += lam_t
            self.c_cumu[np.arange(self.M), arms_t] += c_t[np.arange(self.M), arms_t]
            self.a_con_cumu[np.arange(self.M), arms_t,:] += a_t[np.arange(self.M), arms_t,:]
            self.a_cumu[np.arange(self.M), :,:] += a_t[np.arange(self.M), :,:]
            self.c_hat = np.divide(self.c_cumu, self.pulls,  out=np.zeros_like(self.c_cumu), where=self.pulls != 0)
            self.a_hat = np.divide(self.a_cumu, self.t, out=np.zeros_like(self.a_cumu), where=self.t != 0)
            cons = self.compute_utilization(arms_t)
            cons_total.append(cons)
            result = self.compute_power(stra_init.reshape(self.N,self.M).T,(opt_strategy.reshape(self.N,self.M).T))
            result_total.append(result)

        return result_total, opt_result, cons_total
            
            
#### N: Nodes(N)   M: Functions(M)  K: Resources(k)



