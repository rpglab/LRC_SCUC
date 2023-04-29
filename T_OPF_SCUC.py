"""
Created in 2021

@author: Mingjian Tuo, PhD, University of Houston. 
https://rpglab.github.io/people/Mingjian-Tuo/

Source webpage:
https://rpglab.github.io/resources/LRC-SCUC_Python/

If you use any codes/data here for your work, please cite the following paper:
	Mingjian Tuo and Xingpeng Li, “Security-Constrained Unit Commitment  
	Considering Locational Frequency Stability in Low-Inertia Power Grids”, 
	IEEE Transaction on Power Systems, Oct. 2022.
"""

from pyomo.environ import *
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
#*********************************************************************************************************
# Basic parameter and threshold setting
#*********************************************************************************************************
r0 = 4
sen = 0.96
alpha = -0.5
RoCoFE = 0.5
RoCoF = 0.5
fn = 60
HS = fn*200*2/RoCoF
virtual_price = 0.01
model = AbstractModel()
#*********************************************************************************************************
# Sets
#*********************************************************************************************************
model.BUS = Set()
model.GEND = Set()
model.BRANCH = Set()
model.PERIOD = Set()
#*********************************************************************************************************
# Bus Paramters
#*********************************************************************************************************
model.bus_num = Param(model.BUS)
model.bus_Pd = Param(model.BUS)
model.bus_Solar = Param(model.BUS)
model.bus_Wind = Param(model.BUS)
#*********************************************************************************************************
# Generators Parameters
#*********************************************************************************************************
model.genD_bus = Param(model.GEND)
model.genD_minUP = Param(model.GEND)
model.genD_minDN = Param(model.GEND)
model.genD_status = Param(model.GEND)
model.genD_Pmax = Param(model.GEND)
model.genD_Pmin = Param(model.GEND)
model.genC_Startup = Param(model.GEND)
model.genC_Cost = Param(model.GEND)
model.genC_NLoad = Param(model.GEND)
model.SPRamp = Param(model.GEND)
model.NSRamp = Param(model.GEND)
model.HRamp = Param(model.GEND)
model.StartRamp = Param(model.GEND)
model.gen_Style = Param(model.GEND)
model.inertia_constant = Param(model.GEND)
model.resC = Param(model.GEND)
#*********************************************************************************************************
# Branch Parameters
#*********************************************************************************************************
model.branch_fbus = Param(model.BRANCH)
model.branch_tbus = Param(model.BRANCH)
model.branch_b = Param(model.BRANCH)
model.branch_rateA = Param(model.BRANCH)
model.branch_rateC = Param(model.BRANCH)
model.branch_radial = Param(model.BRANCH)
#*********************************************************************************************************
# Load
#*********************************************************************************************************
model.load_pcnt = Param(model.PERIOD)
model.Wind_pcnt = Param(model.PERIOD)
model.Solar_pcnt = Param(model.PERIOD)
#*********************************************************************************************************
# Variables
#*********************************************************************************************************
model.ug = Var(model.GEND, model.PERIOD, within = Binary)
model.vg = Var(model.GEND, model.PERIOD, within = Binary)
model.theta = Var(model.BUS,model.PERIOD)
model.Pg = Var(model.GEND,model.PERIOD)
model.Pk = Var(model.BRANCH,model.PERIOD)
model.Vi = Var(model.BUS,model.PERIOD)
model.kg = Var(model.GEND, model.PERIOD)
model.re = Var(model.GEND,model.PERIOD)
BaseMVA = 100
#*********************************************************************************************************
# Objective Function and COnstraints Formulations
#*********************************************************************************************************

def obj_cost(model):
    return sum(model.genC_Cost[g]*model.Pg[g,t]*model.ug[g,t]*BaseMVA +  model.genC_Startup[g]*model.vg[g,t] +  model.resC[g]*model.re[g,t]*model.ug[g,t]*BaseMVA for g in model.GEND for t in model.PERIOD) #
    #     + sum(model.Vi[j,t]*virtual_price for j in model.BUS for t in model.PERIOD)
model.obj = Objective(rule=obj_cost,sense=minimize)

# Nodal balance constraints
def power_balance(model,bus,t):
    return -sum(model.Pk[j,t] for j in model.BRANCH if model.branch_fbus[j] == bus) + sum(model.Pk[j,t] for j in model.BRANCH if model.branch_tbus[j] == bus) == \
           -sum(model.Pg[g,t]*model.ug[g,t] for g in model.GEND if model.genD_bus[g] == bus) + model.bus_Pd[bus]*model.load_pcnt[t]/(100*BaseMVA) \
           -sen* model.bus_Wind[bus]*model.Wind_pcnt[t]/100 - r0* model.bus_Solar[bus]*model.Solar_pcnt[t]/100
model.power_balance = Constraint(model.BUS, model.PERIOD, rule=power_balance)

# Generator output power limits
def gen_limit_min(model,j,t):
    return model.Pg[j,t] >= model.genD_Pmin[j]*model.ug[j,t]/BaseMVA
model.gen_min = Constraint(model.GEND,model.PERIOD,rule=gen_limit_min)

def gen_limit_max(model,j,t):
    return model.Pg[j,t]+ model.re[j,t] <= model.genD_Pmax[j]*model.ug[j,t]/BaseMVA
model.gen_max = Constraint(model.GEND,model.PERIOD,rule=gen_limit_max)

# Generator reserve limits
def reserve_limit(model,j,t):
    return 0<=model.re[j,t]<=model.HRamp[j]/BaseMVA
model.reserve = Constraint(model.GEND,model.PERIOD,rule = reserve_limit)

def reserve_sum(model,g,t):
    return sum(model.re[j,t] for j in model.GEND)>=model.Pg[g,t]+ model.re[g,t]
model.sum_re = Constraint(model.GEND,model.PERIOD,rule=reserve_sum)

# Generator ramping limits
def gen_ramping(model,g,t):
    if t >= 2:
        expr = model.Pg[g,t] - model.Pg[g,t-1] - model.HRamp[g]/BaseMVA
        return expr <= 0
    else:
        return Constraint.Skip
model.gen_ramping = Constraint(model.GEND,model.PERIOD,rule=gen_ramping)

def gen_ramping2(model,g,t):
    if t >= 2:
        expr = model.Pg[g,t-1] - model.Pg[g,t] - model.HRamp[g]/BaseMVA
        return expr <= 0
    else:
        return Constraint.Skip
model.gen_ramping2 = Constraint(model.GEND,model.PERIOD,rule=gen_ramping2)

# Line flow constraints
def line_flow(model,j,t):
    return model.Pk[j,t] == -(model.theta[model.branch_fbus[j],t]-model.theta[model.branch_tbus[j],t])*model.branch_b[j]
model.line_flow = Constraint(model.BRANCH,model.PERIOD,rule=line_flow)

# Line thermal constraints
def line_flow_limitslow(model,j,t):
    return -model.branch_rateA[j] <= BaseMVA*model.Pk[j,t]
model.line_flow_limitslow = Constraint(model.BRANCH,model.PERIOD,rule=line_flow_limitslow)

def line_flow_limitshigh(model,j,t):
    return BaseMVA*model.Pk[j,t] <= model.branch_rateA[j]
model.line_flow_limitshigh = Constraint(model.BRANCH,model.PERIOD,rule=line_flow_limitshigh)

# Unit status  constraints
def genUV(model,g,t):
    if t>=2:
        expr = model.ug[g,t] - model.ug[g,t-1] - model.vg[g,t]
        return expr <= 0
    else:
        return Constraint.Skip
model.gen_UV = Constraint(model.GEND,model.PERIOD,rule=genUV)

def geninitial(model,g):
    expr = model.ug[g,1] - model.vg[g,1]
    return expr <= 0
model.Cons_initial = Constraint(model.GEND, rule = geninitial )

'''
# Virtual Inertia/ sync inertia compensation
#def Virtual_Inertia(model,j,t):
#   return model.Vi[j,t]>=0
#model.virtualcon = Constraint(model.BUS,model.PERIOD,rule=Virtual_Inertia)
'''

instance = model.create_instance('./dataFile24BusAllinertia41sen_T.dat')
SCUCsolver = SolverFactory('gurobi')
SCUCsolver.options.mipgap = 0.0001
results = SCUCsolver.solve(instance)
Data, genunit, re = [], [], []
print("\nresults.Solution.Status: " + str(results.Solution.Status))
print("\nresults.solver.status: " + str(results.solver.status))
print("\nresults.solver.termination_condition: " + str(results.solver.termination_condition))
print("\nresults.solver.termination_message: " + str(results.solver.termination_message))
print('\nminimize cost: ' + str(instance.obj()))


for j in instance.PERIOD:
    X = [str(instance.ug[1,j]()),str(instance.ug[2,j]()),str(instance.ug[3,j]()),str(instance.ug[4,j]()),str(instance.ug[5,j]()),str(instance.ug[6,j]()),str(instance.ug[7,j]()),
         str(instance.ug[8,j]()),str(instance.ug[9,j]()),str(instance.ug[10,j]()),str(instance.ug[11,j]()),str(instance.ug[12,j]()),str(instance.ug[13,j]()),str(instance.ug[14,j]()),
         str(instance.ug[15,j]()),str(instance.ug[16,j]()),str(instance.ug[17,j]()),str(instance.ug[18,j]()),str(instance.ug[19,j]()),str(instance.ug[20,j]()),str(instance.ug[21,j]()),
         str(instance.ug[22,j]()),str(instance.ug[23,j]()),str(instance.ug[24,j]()),str(instance.ug[25,j]()),str(instance.ug[26,j]()),str(instance.ug[27,j]()),str(instance.ug[28,j]()),
         str(instance.ug[29,j]()),str(instance.ug[30,j]()),str(instance.ug[31,j]()),str(instance.ug[32,j]()),str(instance.ug[33,j]()),str(instance.ug[34,j]()), str(instance.ug[35,j]()),
         str(instance.ug[36,j]()),str(instance.ug[37,j]()),str(instance.ug[38,j]()),str(instance.ug[39,j]()),str(instance.ug[40,j]()),str(instance.ug[41,j]())]
    Data.append(X)
    Y = [str(instance.Pg[1,j]()),str(instance.Pg[2,j]()),str(instance.Pg[3,j]()),str(instance.Pg[4,j]()),str(instance.Pg[5,j]()),str(instance.Pg[6,j]()),str(instance.Pg[7,j]()),
         str(instance.Pg[8,j]()),str(instance.Pg[9,j]()),str(instance.Pg[10,j]()),str(instance.Pg[11,j]()),str(instance.Pg[12,j]()),str(instance.Pg[13,j]()),str(instance.Pg[14,j]()),
         str(instance.Pg[15,j]()),str(instance.Pg[16,j]()),str(instance.Pg[17,j]()),str(instance.Pg[18,j]()),str(instance.Pg[19,j]()),str(instance.Pg[20,j]()),str(instance.Pg[21,j]()),
         str(instance.Pg[22,j]()),str(instance.Pg[23,j]()),str(instance.Pg[24,j]()),str(instance.Pg[25,j]()),str(instance.Pg[26,j]()),str(instance.Pg[27,j]()),str(instance.Pg[28,j]()),
         str(instance.Pg[29,j]()),str(instance.Pg[30,j]()),str(instance.Pg[31,j]()),str(instance.Pg[32,j]()),str(instance.Pg[33,j]()),str(instance.Pg[34,j]()), str(instance.Pg[35,j]()),
         str(instance.Pg[36,j]()),str(instance.Pg[37,j]()),str(instance.Pg[38,j]()),str(instance.Pg[39,j]()),str(instance.Pg[40,j]()),str(instance.Pg[41,j]())]

    genunit.append(Y)
    Z = [str(instance.re[1,j]()),str(instance.re[2,j]()),str(instance.re[3,j]()),str(instance.re[4,j]()),str(instance.re[5,j]()),str(instance.re[6,j]()),str(instance.re[7,j]()),
         str(instance.re[8,j]()),str(instance.re[9,j]()),str(instance.re[10,j]()),str(instance.re[11,j]()),str(instance.re[12,j]()),str(instance.re[13,j]()),str(instance.re[14,j]()),
         str(instance.re[15,j]()),str(instance.re[16,j]()),str(instance.re[17,j]()),str(instance.re[18,j]()),str(instance.re[19,j]()),str(instance.re[20,j]()),str(instance.re[21,j]()),
         str(instance.re[22,j]()),str(instance.re[23,j]()),str(instance.re[24,j]()),str(instance.re[25,j]()),str(instance.re[26,j]()),str(instance.re[27,j]()),str(instance.re[28,j]()),
         str(instance.re[29,j]()),str(instance.re[30,j]()),str(instance.re[31,j]()),str(instance.re[32,j]()),str(instance.re[33,j]()),str(instance.re[34,j]()), str(instance.re[35,j]()),
         str(instance.re[36,j]()),str(instance.re[37,j]()),str(instance.re[38,j]()),str(instance.re[39,j]()),str(instance.re[40,j]()),str(instance.re[41,j]())]
    re.append(Z)

Data =pd.DataFrame(Data,columns=['pg1','pg2','pg3','pg4','pg5','pg6','pg7','pg8','pg9','pg10',
                                 'pg11','pg12','pg13','pg14','pg15','pg16','pg17','pg18','pg19',
                                 'pg20','pg21','pg22','pg23','pg24','pg25','pg26','pg27','pg28',
                                 'pg29','pg30','pg31','pg32','pg33','pg34','pg35','pg36','pg37',
                                 'pg38','pg39','pg40','pg41'])
genunit =pd.DataFrame(genunit,columns=['pg1','pg2','pg3','pg4','pg5','pg6','pg7','pg8','pg9','pg10',
                                       'pg11','pg12','pg13','pg14','pg15','pg16','pg17','pg18','pg19',
                                       'pg20','pg21','pg22','pg23','pg24','pg25','pg26','pg27','pg28',
                                       'pg29','pg30','pg31','pg32','pg33','pg34','pg35','pg36','pg37',
                                       'pg38','pg39','pg40','pg41'])
re = pd.DataFrame(re,columns=['pg1','pg2','pg3','pg4','pg5','pg6','pg7','pg8','pg9','pg10',
                                       'pg11','pg12','pg13','pg14','pg15','pg16','pg17','pg18','pg19',
                                       'pg20','pg21','pg22','pg23','pg24','pg25','pg26','pg27','pg28',
                                       'pg29','pg30','pg31','pg32','pg33','pg34','pg35','pg36','pg37',
                                       'pg38','pg39','pg40','pg41'])

writer_1 = pd.ExcelWriter('./T_Data_Pg.xlsx')
genunit.to_excel(writer_1, index=False,encoding='utf-8',sheet_name='Sheet')
writer_1.save()
writer_2 = pd.ExcelWriter('./T_Data_UC.xlsx')
Data.to_excel(writer_2, index=False,encoding='utf-8',sheet_name='Sheet')
writer_2.save()
writer_3 = pd.ExcelWriter('./T_Data_RE.xlsx')
re.to_excel(writer_3, index=False,encoding='utf-8',sheet_name='Sheet')
writer_3.save()
