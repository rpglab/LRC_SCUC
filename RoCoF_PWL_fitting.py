"""
Created in 2021

@author: Mingjian Tuo, PhD, University of Houston. 
https://rpglab.github.io/people/Mingjian-Tuo/

Source webpage:
https://rpglab.github.io/resources/LRC-SCUC_Python/

If you use any codes/data here for your work, please cite the following paper:
	Mingjian Tuo and Xingpeng Li, “Security-Constrained Unit Commitment Considering Locational Frequency Stability in Low-Inertia Power Grids”, IEEE Transaction on Power Systems, Oct. 2022.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pyomo.environ import *
import os
from mpl_toolkits.mplot3d import Axes3D
import cplex
plt.rc('font', family = 'Times New Roman')

## ****************************************************************************
##						  Basic Parameters and Reduced System Data
## ****************************************************************************
v2 = [0.3066, 0.3072,0.6571, 0.0656, -0.2376, -0.16, -0.27, -0.30, -0.35, -0.005] # Example of eigenvector values of erduced model
nim = 4.88
Tw = 0.1 # average over Tw period
wn = 2 * np.pi * 60
r = 0.1 # approximation ratio of damping to inertia
T = 0 # start measuring time

# definition of RoCoF calculation functions

def ROCOF0(d, M, ml):
    expr = d*(1 - np.exp(- r * Tw))/(2  *np.pi * r * Tw * (M - ml))
    return expr

def ROCOF(d, M, ml,a, b):
    expr = d * (1 - np.exp(-r * Tw))/(2*np.pi * r * Tw * (M - ml)) + v2[a]*v2[b]*d*np.exp(- r * Tw/2)*np.sin(np.sqrt(nim/(M/10-ml/10)-r*r/4)*Tw)/((2 * np.pi * 2*(M-ml)/10) * np.sqrt(nim/(M/10-ml/10)-r*r/4)*Tw)
    return expr

def ROCOF_T(d, M, ml,a,b):
    base = d*np.exp(- r * T) * (1 - np.exp(-r*Tw))/(2 * np.pi * r * Tw * (M-ml))
    osc2 = np.exp(- r * T/2) * v2[a] * v2[b] * d
    nom2 = (np.exp(- r * Tw/2) * np.sin(np.sqrt(nim/(M/10 - ml/10) - r*r/4) * (T + Tw)) - np.sin(np.sqrt(nim/(M/10 - ml/10) - r * r/4) * T))
    denom2 = (2 * np.pi * (M) / 10 * np.sqrt(nim / (M/10 - ml/10) - r * r / 4) * Tw)
    #osc3 = np.exp(-r*T/2)*v3[a]*v3[b]*d
    #nom3 = (np.exp(-r*Tw/2)*np.sin(np.sqrt(nim3/(M/10-ml/10)-r*r/4)*(T+Tw))-np.sin(np.sqrt(nim3/(M/10-ml/10)-r*r/4)*T))
    #denom3 = (2 * np.pi * (M) / 10 * np.sqrt(nim3 / (M/10-ml/10) - r * r / 4) * Tw)
    #osc4 = np.exp(-r*T/2)*v4[a]*v4[b]*d
    #nom4 = (np.exp(-r*Tw/2)*np.sin(np.sqrt(nim4/(M/10-ml/10)-r*r/4)*(T+Tw))-np.sin(np.sqrt(nim4/(M/10-ml/10)-r*r/4)*T))
    #denom4 = (2 * np.pi * (M) / 10 * np.sqrt(nim4 / (M/10-ml/10) - r * r / 4) * Tw)
    expr =base +osc2*nom2/denom2  #+ osc3*nom3/denom3 + osc4*nom4/denom4
    return expr

model = AbstractModel()

## ****************************************************************************
##						  Sets
## ****************************************************************************
model.SEG = Set()
model.EP = Set()

## ****************************************************************************
##						  Variables
## ****************************************************************************
model.t = Var(model.SEG,model.EP)
model.v = Var( model.SEG,model.EP, within = Binary)
model.a = Var(model.SEG)
model.b = Var(model.SEG)
model.c = Var(model.SEG)
model.d = Var(model.SEG)

## ****************************************************************************
##						  Parameters
## ****************************************************************************
model.dis = Param(model.EP)
model.m = Param(model.EP)
model.dm = Param(model.EP)
model.num = Param(model.SEG)
A = 10000 # big constant number

def obj_cost(model):
    return sum((model.t[3,i] - ROCOF_T(model.dis[i],model.m[i],model.dm[i],8,9))**2 for i in model.EP)
model.obj = Objective(rule=obj_cost,sense=minimize)

def l1(model,i):
    return model.a[1]*model.dis[i] + model.b[1]*model.m[i] + model.c[1]*model.dm[i] + model.d[1] <= model.t[1,i]
model.l1 = Constraint(model.EP, rule = l1)

def u1(model,i):
    return model.a[1]*model.dis[i] + model.b[1]*model.m[i] + model.c[1]*model.dm[i] + model.d[1] + model.v[1,i]*A >= model.t[1,i]
model.u1 = Constraint(model.EP, rule = u1)

def l2(model,i):
    return model.a[2]*model.dis[i] + model.b[2]*model.m[i] + model.c[2]*model.dm[i] + model.d[2] <= model.t[1,i]
model.l2 = Constraint(model.EP,rule = l2)

def u2(model,i):
    return model.a[2]*model.dis[i] + model.b[2]*model.m[i] + model.c[2]*model.dm[i] + model.d[2] + (1 - model.v[1,i])*A >= model.t[1,i]
model.u2 = Constraint(model.EP,rule = u2)

def t12(model,i):
    return model.t[1,i] <= model.t[2,i]
model.t12 = Constraint(model.EP, rule = t12)

def t21(model,i):
    return model.t[2,i]<=model.t[1,i] + model.v[2,i]*A
model.t21 = Constraint(model.EP,rule = t21)

def l3(model,i):
    return model.a[3]*model.dis[i] + model.b[3]*model.m[i] + model.c[3]*model.dm[i] + model.d[3] <= model.t[2,i]
model.l3 = Constraint(model.EP,rule = l3)

def u3(model,i):
    return model.a[3]*model.dis[i] + model.b[3]*model.m[i] + model.c[3]*model.dm[i] + model.d[3] + (1 - model.v[2,i])*A >= model.t[2,i]
model.u3 = Constraint(model.EP,rule = u3)

def t23(model,i):
    return model.t[2,i] <= model.t[3,i]
model.t23 = Constraint(model.EP,rule = t23)

def t32(model,i):
    return model.t[3,i]<=model.t[2,i] + model.v[3,i]*A
model.t32 = Constraint(model.EP,rule = t32)

def l4(model,i):
    return model.a[4]*model.dis[i] + model.b[4]*model.m[i] + model.c[4]*model.dm[i] + model.d[4] <= model.t[3,i]
model.l4 = Constraint(model.EP,rule = l4)

def u4(model,i):
    return model.a[4]*model.dis[i] + model.b[4]*model.m[i] + model.c[4]*model.dm[i] + model.d[4] + (1 - model.v[3,i])*A >= model.t[3,i]
model.u4 = Constraint(model.EP,rule = u4)

def coeff_a(model,s):
    if s >= 2:
        return model.a[s] >= model.a[s-1]+0.00025
    else:
        return Constraint.Skip
model.coeff_a = Constraint(model.SEG,rule = coeff_a)

def coeff_d_up(model,s):
        return model.d[s] <= 0.5
model.coeff_d_up = Constraint(model.SEG,rule = coeff_d_up)

def coeff_d_low(model, s):
    return model.d[s] >= 0
model.coeff_d_low = Constraint(model.SEG, rule=coeff_d_low)

def coeff_a_up(model, s):
    return model.a[s] <= 10
model.coeff_a_up = Constraint(model.SEG, rule=coeff_a_up)


def coeff_a_low(model, s):
    return model.a[s] >= -10
model.coeff_a_low = Constraint(model.SEG, rule=coeff_a_low)

def coeff_b_up(model, s):
    return model.b[s] <= 1
model.coeff_b_up = Constraint(model.SEG, rule=coeff_b_up)


def coeff_b_low(model, s):
    return model.b[s] >= -1
model.coeff_b_low = Constraint(model.SEG, rule=coeff_b_low)

def coeff_b(model,s):
    if s >= 2:
        return model.b[s] >= model.b[s-1]+0.00025
    else:
        return Constraint.Skip
model.coeff_b = Constraint(model.SEG,rule = coeff_b)

def coeff_c(model,s):
        return model.c[s] == 0
model.coeff_c = Constraint(model.SEG,rule = coeff_c)

instance = model.create_instance('./data_PWL_4EP.dat')
solver = SolverFactory('copt',executable = 'D:/ampl_mswin64/ampl_mswin64/copt')
solver.options.mipgap = 0.0001
results = solver.solve(instance)
print("\nresults.solver.termination_condition: " + str(results.solver.termination_condition))
print("\nresults.solver.termination_message: " + str(results.solver.termination_message))
print('\nminimize cost: ' + str(instance.obj()))

import pandas as pd
pd.set_option('display.max_columns', None)
coe = []
for j in instance.SEG:
    X = [str(instance.a[j]()),str(instance.b[j]()),str(instance.c[j]()),str(instance.d[j]())]
    coe.append(X)
COE =pd.DataFrame(coe,columns=['a','b','c','d'])

'''
k1, k2, k3, k4 = [], [], [], []

for j in instance.seg:
    X = [str(instance.a[j]()), str(instance.a[j]()), str(instance.a[j]()), str(instance.a[j]())]
    k1.append(X)
    Y = [str(instance.b[j]()), str(instance.b[j]()), str(instance.b[j]()), str(instance.b[j]())]
    k2.append(Y)
    Z = [str(instance.c[j]()), str(instance.c[j]()), str(instance.c[j]()), str(instance.c[j]())]
    k3.append(Z)
    R = [str(instance.d[j]()), str(instance.d[j]()), str(instance.d[j]()), str(instance.d[j]())]
    k4.append(X)

x1, x2 = np.meshgrid(x1, x2)
y0 = ROCOF(x1,x2,x3)
y1 = ROCOF1(x1,x2,x3)
figure = plt.figure()
ax = Axes3D(figure)
surface = ax.plot_surface(x1,x2*wn,y0,cmap = 'rainbow',antialiased=False)
figure.colorbar(surface, shrink=0.8)
ax.view_init(25, 40)

ax.plot_surface(x1,x2,y0,rstride=1,cstride=1,cmap='rainbow')
#ax.plot_surface(x1,x2,y1,rstride=1,cstride=1,cmap='Greens')
ax.set_xlabel('/Delta P (MW)')
ax.set_ylabel('H (MWs)')
ax.set_zlabel('RoCoF (HZ/s)')
ax.w_xaxis.set_pane_color((1.0,1.0,1.0))
ax.w_yaxis.set_pane_color((1.0,1.0,1.0))
ax.w_zaxis.set_pane_color((1.0,1.0,1.0))
plt.show()
print(y0)
'''

writer_1 = pd.ExcelWriter('./COE_Files/pwl_coefficients_busXX.xlsx')
COE.to_excel(writer_1, index=False,encoding='utf-8',sheet_name='Sheet')
writer_1.save()