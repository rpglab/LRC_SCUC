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
# define coefficient values root
#*********************************************************************************************************
COE_file_root_bus01 ="./COE_Files/EP4_pwl_coefficients_bus01.xlsx"
COE_file_root_bus02 ="./COE_Files/EP4_pwl_coefficients_bus02.xlsx"
COE_file_root_bus07 ="./COE_Files/EP4_pwl_coefficients_bus07.xlsx"
COE_file_root_bus13 ="./COE_Files/EP4_pwl_coefficients_bus13.xlsx"
COE_file_root_bus15 ="./COE_Files/EP4_pwl_coefficients_bus15.xlsx"
COE_file_root_bus16 ="./COE_Files/EP4_pwl_coefficients_bus16.xlsx"
COE_file_root_bus18 ="./COE_Files/EP4_pwl_coefficients_bus18.xlsx"
COE_file_root_bus21 ="./COE_Files/EP4_pwl_coefficients_bus21.xlsx"
COE_file_root_bus22 ="./COE_Files/EP4_pwl_coefficients_bus22.xlsx"
COE_file_root_bus23 ="./COE_Files/EP4_pwl_coefficients_bus23.xlsx"
COE_file_root_bus01_t2 ="./COE_Files/EP4_pwl_coefficients_bus01_t2.xlsx"
COE_file_root_bus02_t2 ="./COE_Files/EP4_pwl_coefficients_bus02_t2.xlsx"
COE_file_root_bus07_t2 ="./COE_Files/EP4_pwl_coefficients_bus07_t2.xlsx"
COE_file_root_bus13_t2 ="./COE_Files/EP4_pwl_coefficients_bus13_t2.xlsx"
COE_file_root_bus15_t2 ="./COE_Files/EP4_pwl_coefficients_bus15_t2.xlsx"
COE_file_root_bus16_t2 ="./COE_Files/EP4_pwl_coefficients_bus16_t2.xlsx"
COE_file_root_bus18_t2 ="./COE_Files/EP4_pwl_coefficients_bus18_t2.xlsx"
COE_file_root_bus21_t2 ="./COE_Files/EP4_pwl_coefficients_bus21_t2.xlsx"
COE_file_root_bus22_t2 ="./COE_Files/EP4_pwl_coefficients_bus22_t2.xlsx"
COE_file_root_bus23_t2 ="./COE_Files/EP4_pwl_coefficients_bus23_t2.xlsx"

# read coefficient values
def Read_Coe(link):
    df = pd.read_excel(link)
    seg1 = df.values[0,:]
    seg2 = df.values[1,:]
    seg3 = df.values[2,:]
    seg4 = df.values[3,:]
    return seg1, seg2, seg3, seg4

bus1_seg1,bus1_seg2,bus1_seg3,bus1_seg4 = Read_Coe(COE_file_root_bus01)
bus2_seg1,bus2_seg2,bus2_seg3,bus2_seg4 = Read_Coe(COE_file_root_bus02)
bus7_seg1,bus7_seg2,bus7_seg3,bus7_seg4 = Read_Coe(COE_file_root_bus07)
bus13_seg1,bus13_seg2,bus13_seg3,bus13_seg4 = Read_Coe(COE_file_root_bus13)
bus15_seg1,bus15_seg2,bus15_seg3,bus15_seg4 = Read_Coe(COE_file_root_bus15)
bus16_seg1,bus16_seg2,bus16_seg3,bus16_seg4 = Read_Coe(COE_file_root_bus16)
bus18_seg1,bus18_seg2,bus18_seg3,bus18_seg4 = Read_Coe(COE_file_root_bus18)
bus21_seg1,bus21_seg2,bus21_seg3,bus21_seg4 = Read_Coe(COE_file_root_bus21)
bus22_seg1,bus22_seg2,bus22_seg3,bus22_seg4 = Read_Coe(COE_file_root_bus22)
bus23_seg1,bus23_seg2,bus23_seg3,bus23_seg4 = Read_Coe(COE_file_root_bus23)
bus1_seg1_t2,bus1_seg2_t2,bus1_seg3_t2,bus1_seg4_t2 = Read_Coe(COE_file_root_bus01_t2)
bus2_seg1_t2,bus2_seg2_t2,bus2_seg3_t2,bus2_seg4_t2 = Read_Coe(COE_file_root_bus02_t2)
bus7_seg1_t2,bus7_seg2_t2,bus7_seg3_t2,bus7_seg4_t2 = Read_Coe(COE_file_root_bus07_t2)
bus13_seg1_t2,bus13_seg2_t2,bus13_seg3_t2,bus13_seg4_t2 = Read_Coe(COE_file_root_bus13_t2)
bus15_seg1_t2,bus15_seg2_t2,bus15_seg3_t2,bus15_seg4_t2 = Read_Coe(COE_file_root_bus15_t2)
bus16_seg1_t2,bus16_seg2_t2,bus16_seg3_t2,bus16_seg4_t2 = Read_Coe(COE_file_root_bus16_t2)
bus18_seg1_t2,bus18_seg2_t2,bus18_seg3_t2,bus18_seg4_t2 = Read_Coe(COE_file_root_bus18_t2)
bus21_seg1_t2,bus21_seg2_t2,bus21_seg3_t2,bus21_seg4_t2 = Read_Coe(COE_file_root_bus21_t2)
bus22_seg1_t2,bus22_seg2_t2,bus22_seg3_t2,bus22_seg4_t2 = Read_Coe(COE_file_root_bus22_t2)
bus23_seg1_t2,bus23_seg2_t2,bus23_seg3_t2,bus23_seg4_t2 = Read_Coe(COE_file_root_bus23_t2)

#*********************************************************************************************************
# Parameters Setting
#*********************************************************************************************************
r0 = 1
sen = 1
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
# Load Profiles
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
# PWL of nodal equations
#*********************************************************************************************************
model.SEG = Set()
model.t01 = Var(model.SEG,model.PERIOD)
model.v01 = Var(model.SEG,model.PERIOD, within = Binary)
model.t02_1 = Var(model.SEG,model.PERIOD)
model.v02_1 = Var(model.SEG,model.PERIOD, within = Binary)
model.t02_2 = Var(model.SEG,model.PERIOD)
model.v02_2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t02_3 = Var(model.SEG,model.PERIOD)
model.v02_3 = Var(model.SEG,model.PERIOD, within = Binary)
model.t02_4 = Var(model.SEG,model.PERIOD)
model.v02_4 = Var(model.SEG,model.PERIOD, within = Binary)
model.t02_5 = Var(model.SEG,model.PERIOD)
model.v02_5 = Var(model.SEG,model.PERIOD, within = Binary)
model.t02_6 = Var(model.SEG,model.PERIOD)
model.v02_6 = Var(model.SEG,model.PERIOD, within = Binary)
model.t07_1 = Var(model.SEG,model.PERIOD)
model.v07_1 = Var(model.SEG,model.PERIOD, within = Binary)
model.t07_2 = Var(model.SEG,model.PERIOD)
model.v07_2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t07_3 = Var(model.SEG,model.PERIOD)
model.v07_3 = Var(model.SEG,model.PERIOD, within = Binary)
model.t13_1 = Var(model.SEG,model.PERIOD)
model.v13_1 = Var(model.SEG,model.PERIOD, within = Binary)
model.t13_2 = Var(model.SEG,model.PERIOD)
model.v13_2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t13_3 = Var(model.SEG,model.PERIOD)
model.v13_3 = Var(model.SEG,model.PERIOD, within = Binary)
model.t13_4 = Var(model.SEG,model.PERIOD)
model.v13_4 = Var(model.SEG,model.PERIOD, within = Binary)
model.t13_5 = Var(model.SEG,model.PERIOD)
model.v13_5 = Var(model.SEG,model.PERIOD, within = Binary)
model.t13_6 = Var(model.SEG,model.PERIOD)
model.v13_6 = Var(model.SEG,model.PERIOD, within = Binary)
model.t15 = Var(model.SEG,model.PERIOD)
model.v15 = Var(model.SEG,model.PERIOD, within = Binary)
model.t15_1 = Var(model.SEG,model.PERIOD)
model.v15_1 = Var(model.SEG,model.PERIOD, within = Binary)
model.t16 = Var(model.SEG,model.PERIOD)
model.v16 = Var(model.SEG,model.PERIOD, within = Binary)
model.t18 = Var(model.SEG,model.PERIOD)
model.v18 = Var(model.SEG,model.PERIOD, within = Binary)
model.t21 = Var(model.SEG,model.PERIOD)
model.v21 = Var(model.SEG,model.PERIOD, within = Binary)
model.num = Param(model.SEG)
model.t22 = Var(model.SEG,model.PERIOD)
model.v22 = Var(model.SEG,model.PERIOD, within = Binary)
model.t23 = Var(model.SEG,model.PERIOD)
model.v23 = Var(model.SEG,model.PERIOD, within = Binary)
#*********************************************************************************************************
# Second  Measuring Window Variables
#*********************************************************************************************************
model.t01_t2 = Var(model.SEG,model.PERIOD)
model.v01_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t02_1_t2 = Var(model.SEG,model.PERIOD)
model.v02_1_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t02_2_t2 = Var(model.SEG,model.PERIOD)
model.v02_2_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t02_3_t2 = Var(model.SEG,model.PERIOD)
model.v02_3_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t02_4_t2 = Var(model.SEG,model.PERIOD)
model.v02_4_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t02_5_t2 = Var(model.SEG,model.PERIOD)
model.v02_5_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t02_6_t2 = Var(model.SEG,model.PERIOD)
model.v02_6_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t07_1_t2 = Var(model.SEG,model.PERIOD)
model.v07_1_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t07_2_t2 = Var(model.SEG,model.PERIOD)
model.v07_2_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t07_3_t2 = Var(model.SEG,model.PERIOD)
model.v07_3_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t13_1_t2 = Var(model.SEG,model.PERIOD)
model.v13_1_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t13_2_t2 = Var(model.SEG,model.PERIOD)
model.v13_2_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t13_3_t2 = Var(model.SEG,model.PERIOD)
model.v13_3_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t13_4_t2 = Var(model.SEG,model.PERIOD)
model.v13_4_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t13_5_t2 = Var(model.SEG,model.PERIOD)
model.v13_5_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t13_6_t2 = Var(model.SEG,model.PERIOD)
model.v13_6_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t15_t2 = Var(model.SEG,model.PERIOD)
model.v15_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t15_1_t2 = Var(model.SEG,model.PERIOD)
model.v15_1_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t16_t2 = Var(model.SEG,model.PERIOD)
model.v16_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t18_t2 = Var(model.SEG,model.PERIOD)
model.v18_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t21_t2 = Var(model.SEG,model.PERIOD)
model.v21_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t22_t2 = Var(model.SEG,model.PERIOD)
model.v22_t2 = Var(model.SEG,model.PERIOD, within = Binary)
model.t23_t2 = Var(model.SEG,model.PERIOD)
model.v23_t2 = Var(model.SEG,model.PERIOD, within = Binary)
A =1000 # Big Number Method
#*********************************************************************************************************
# Objective Function and Constraints Formulations
#*********************************************************************************************************
def obj_cost(model):
    return sum(model.genC_Cost[g]*model.Pg[g,t]*model.ug[g,t]*BaseMVA + model.genC_Startup[g]*model.vg[g,t] + model.resC[g] * model.re[g,t]*model.ug[g,t]*BaseMVA for g in model.GEND for t in model.PERIOD) #
    #     + sum(model.Vi[j,t]*virtual_price for j in model.BUS for t in model.PERIOD)
model.obj = Objective(rule=obj_cost,sense=minimize)

# Nodal balance constraints
def power_balance(model,bus,t):
    return - sum(model.Pk[j,t] for j in model.BRANCH if model.branch_fbus[j] == bus) + sum(model.Pk[j,t] for j in model.BRANCH if model.branch_tbus[j] == bus) == \
           - sum(model.Pg[g,t] * model.ug[g,t] for g in model.GEND if model.genD_bus[g] == bus) + model.bus_Pd[bus]*model.load_pcnt[t]/(100*BaseMVA) \
           - sen* model.bus_Wind[bus]*model.Wind_pcnt[t]/100 - r0 * model.bus_Solar[bus] * model.Solar_pcnt[t]/100
model.power_balance = Constraint(model.BUS, model.PERIOD, rule=power_balance)

def gen_limit_min(model,j,t):
    return model.Pg[j,t] >= model.genD_Pmin[j]*model.ug[j,t]/BaseMVA
model.gen_min = Constraint(model.GEND,model.PERIOD,rule=gen_limit_min)

def gen_limit_max(model,j,t):
    return model.Pg[j,t]+ model.re[j,t] <= model.genD_Pmax[j]*model.ug[j,t]/BaseMVA
model.gen_max = Constraint(model.GEND,model.PERIOD,rule=gen_limit_max)

def reserve_limit(model,j,t):
    return 0<=model.re[j,t]<=model.HRamp[j]/BaseMVA
model.reserve = Constraint(model.GEND,model.PERIOD,rule = reserve_limit)

def reserve_sum(model,g,t):
    return sum(model.re[j,t] for j in model.GEND)>=model.Pg[g,t]+ model.re[g,t]
model.sum_re = Constraint(model.GEND,model.PERIOD,rule=reserve_sum)

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

# Line flow
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

# ERC-RoCoF constraints
'''
# ERC-SCUC Model
def minimal_inertia(model,g,j,t):
    expr =  sum(model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t] for g in model.GEND) \
            - model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t] >= 100*model.Pg[g,t]*model.ug[g,t]*fn/(2*RoCoFE)
    return expr
# + sum(model.Vi[j, t] for j in model.BUS)\
model.inertia_constraint = Constraint(model.GEND,model.BUS,model.PERIOD,rule = minimal_inertia)
'''
# Unit satus  constraints
def genUV(model,g,t):
    if t>=2:
        expr = model.ug[g,t] - model.ug[g,t-1] - model.vg[g,t]
        return expr <= 0
    else:
        return Constraint.Skip
model.gen_UV = Constraint(model.GEND,model.PERIOD,rule=genUV)

# Initialize constraint
def geninitial(model,g):
    expr = model.ug[g,1] - model.vg[g,1]
    return expr <= 0
model.Cons_initial = Constraint(model.GEND, rule = geninitial )

#*********************************************************************************************************
# PWL nodal RoCoF Constraints
#*********************************************************************************************************
# Node 01 locational RoCoF constraints

def bus01_R(model,t):
    return model.t01[3,t] <= RoCoF
model.bus01_R = Constraint(model.PERIOD,rule = bus01_R)

def bus01_l1(model,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg1[0]*100*model.Pg[g,t]*model.ug[g,t]+ bus1_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND)) + bus1_seg1[2] * model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t] + bus1_seg1[3] <= model.t01[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus01_l1 = Constraint(model.GEND,model.PERIOD,rule = bus01_l1)

def bus01_u1(model,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg1[0]*100*model.Pg[g,t]*model.ug[g,t] + bus1_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus1_seg1[2] * model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) + bus1_seg1[3] + model.v01[1,t]*A >= model.t01[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus01_u1 = Constraint(model.GEND, model.PERIOD, rule = bus01_u1)

def bus01_l2(model,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus1_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND)) + bus1_seg2[2] * model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t] +  bus1_seg2[3] <= model.t01[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus01_l2 = Constraint(model.GEND, model.PERIOD,rule = bus01_l2)

def bus01_u2(model,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg2[0]*100*model.Pg[g,t]*model.ug[g,t] +  bus1_seg2[1]/(fn*np.pi)*(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND)) + bus1_seg2[2] * model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]+  bus1_seg2[3] + ( 1 - model.v01[1, t]) * A >= model.t01[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus01_u2 = Constraint(model.GEND, model.PERIOD, rule = bus01_u2)

def bus01_t12(model,t):
    return model.t01[1,t] <= model.t01[2,t]
model.bus01_t12 = Constraint( model.PERIOD,rule = bus01_t12)

def bus01_t21(model,t):
    return model.t01[2,t]<=model.t01[1,t] + model.v01[2,t]*A
model.bus01_t21 = Constraint(model.PERIOD, rule = bus01_t21)

def bus01_l3(model,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg3[0]*100*model.Pg[g,t]*model.ug[g,t] + bus1_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus1_seg3[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus1_seg3[3]<= model.t01[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus01_l3 = Constraint(model.GEND,model.PERIOD, rule = bus01_l3)

def bus01_u3(model,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg3[0]*100*model.Pg[g,t]*model.ug[g,t] + bus1_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND)) + bus1_seg3[2]* model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]  + bus1_seg3[3] + (1 - model.v01[2,t])*A >= model.t01[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus01_u3 = Constraint(model.GEND, model.PERIOD,rule = bus01_u3)

def bus01_t23(model,t):
    return model.t01[2,t] <= model.t01[3,t]
model.bus01_t23 = Constraint(model.PERIOD, rule = bus01_t23)

def bus01_t32(model,t):
    return model.t01[3,t]<=model.t01[2,t] + model.v01[3,t]*A
model.bus01_t32 = Constraint(model.PERIOD,rule = bus01_t32)

def bus01_l4(model,g,t):
    if g >= 1:
        if g <= 4:
             return bus1_seg4[0]*model.Pg[g,t]*model.ug[g,t] + bus1_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND)) + bus1_seg4[2]* model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t] + bus1_seg4[3] <= model.t01[3,t]
        else:
             return Constraint.Skip
    else:
        return Constraint.Skip
model.bus01_l4 = Constraint(model.GEND,model.PERIOD,rule = bus01_l4)

def bus01_u4(model,g,t):
    if g >= 1:
        if g <= 4:
             return bus1_seg4[0]*model.Pg[g,t]*model.ug[g,t] + bus1_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND)) + bus1_seg4[2]* model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t] + bus1_seg4[3] +(1 - model.v01[3,t])*A >= model.t01[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus01_u4 = Constraint(model.GEND,model.PERIOD,rule = bus01_u4)

# Node 02 locational RoCoF constraints

def bus02_1_R(model,t):
    return model.t02_1[3,t] <= RoCoF
model.bus02_1_R = Constraint(model.PERIOD, rule = bus02_1_R)

def bus02_1_l1(model,t):
    return bus2_seg1[0]*100*model.Pg[5,t]*model.ug[5,t] + bus2_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1[2] * model.inertia_constant[5]*model.genD_Pmax[5]*model.ug[5,t]) + bus2_seg1[3] <= model.t02_1[1,t]
model.bus02_1_l1 = Constraint(model.PERIOD, rule = bus02_1_l1)

def bus02_1_u1(model,t):
    return bus2_seg1[0]*100*model.Pg[5,t]*model.ug[5,t] + bus2_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1[2] * model.inertia_constant[5]*model.genD_Pmax[5]*model.ug[5,t]) + bus2_seg1[3] + model.v02_1[1,t]*A >= model.t02_1[1,t]
model.bus02_1_u1 = Constraint(model.PERIOD, rule = bus02_1_u1)

def bus02_1_l2(model,t):
    return bus2_seg2[0]*100*model.Pg[5,t]*model.ug[5,t] + bus2_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2[2] * model.inertia_constant[5]*model.genD_Pmax[5]*model.ug[5,t]) + bus2_seg2[3] <= model.t02_1[1,t]
model.bus02_1_l2 = Constraint(model.PERIOD,rule = bus02_1_l2)

def bus02_1_u2(model,t):
    return bus2_seg2[0]*100*model.Pg[5,t]*model.ug[5,t] + bus2_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2[2] * model.inertia_constant[5]*model.genD_Pmax[5]*model.ug[5,t]) + bus2_seg2[3] + (1 - model.v02_1[1,t])*A >= model.t02_1[1,t]
model.bus02_1_u2 = Constraint(model.PERIOD,rule = bus02_1_u2)

def bus02_1_t12(model,t):
    return model.t02_1[1,t] <= model.t02_1[2,t]
model.bus02_1_t12 = Constraint(model.PERIOD,rule = bus02_1_t12)

def bus02_1_t21(model,t):
    return model.t02_1[2,t]<=model.t02_1[1,t] + model.v02_1[2,t]*A
model.bus02_1_t21 = Constraint(model.PERIOD,rule = bus02_1_t21)

def bus02_1_l3(model,t):
    return bus2_seg3[0]*100*model.Pg[5,t]*model.ug[5,t] + bus2_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3[2]* model.inertia_constant[5]*model.genD_Pmax[5]*model.ug[5,t]) + bus2_seg3[3]  <= model.t02_1[2,t]
model.bus02_1_l3 = Constraint(model.PERIOD,rule = bus02_1_l3)

def bus02_1_u3(model,t):
    return bus2_seg3[0]*100*model.Pg[5,t]*model.ug[5,t] + bus2_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3[2]* model.inertia_constant[5]*model.genD_Pmax[5]*model.ug[5,t]) + bus2_seg3[3]   + (1 - model.v02_1[2,t])*A >= model.t02_1[2,t]
model.bus02_1_u3 = Constraint(model.PERIOD,rule = bus02_1_u3)

def bus02_1_t23(model,t):
    return model.t02_1[2,t] <= model.t02_1[3,t]
model.bus02_1_t23 = Constraint(model.PERIOD,rule = bus02_1_t23)

def bus02_1_t32(model,t):
    return model.t02_1[3,t]<=model.t02_1[2,t] + model.v02_1[3,t]*A
model.bus02_1_t32 = Constraint(model.PERIOD,rule = bus02_1_t32)

def bus02_1_l4(model,t):
    return bus2_seg4[0]*100*model.Pg[5,t]*model.ug[5,t] + bus2_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4[2]* model.inertia_constant[5]*model.genD_Pmax[5]*model.ug[5,t]) + bus2_seg4[3]  <= model.t02_1[3,t]
model.bus02_1_l4 = Constraint(model.PERIOD,rule = bus02_1_l4)

def bus02_1_u4(model,t):
    return bus2_seg4[0]*100*model.Pg[5,t]*model.ug[5,t] + bus2_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4[2]* model.inertia_constant[5]*model.genD_Pmax[5]*model.ug[5,t]) + bus2_seg4[3]+(1 - model.v02_1[3,t])*A >= model.t02_1[3,t]
model.bus02_1_u4 = Constraint(model.PERIOD,rule = bus02_1_u4)

def bus02_2_R(model,t):
    return model.t02_2[3,t] <= RoCoF
model.bus02_2_R = Constraint(model.PERIOD, rule = bus02_2_R)

def bus02_2_l1(model,t):
    return bus2_seg1[0]*100*model.Pg[6,t]*model.ug[6,t] + bus2_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1[2]* model.inertia_constant[6]*model.genD_Pmax[6]*model.ug[6,t]) + bus2_seg1[3]<= model.t02_2[1,t]
model.bus02_2_l1 = Constraint(model.PERIOD, rule = bus02_2_l1)

def bus02_2_u1(model,t):
    return bus2_seg1[0]*100*model.Pg[6,t]*model.ug[6,t] + bus2_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1[2]* model.inertia_constant[6]*model.genD_Pmax[6]*model.ug[6,t]) + bus2_seg1[3] + model.v02_2[1,t]*A >= model.t02_2[1,t]
model.bus02_2_u1 = Constraint(model.PERIOD, rule = bus02_2_u1)

def bus02_2_l2(model,t):
    return bus2_seg2[0]*100*model.Pg[6,t]*model.ug[6,t] + bus2_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2[2]* model.inertia_constant[6]*model.genD_Pmax[6]*model.ug[6,t]) + bus2_seg2[3] <= model.t02_2[1,t]
model.bus02_2_l2 = Constraint(model.PERIOD,rule = bus02_2_l2)

def bus02_2_u2(model,t):
    return bus2_seg2[0]*100*model.Pg[6,t]*model.ug[6,t] + bus2_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2[2]* model.inertia_constant[6]*model.genD_Pmax[6]*model.ug[6,t]) + bus2_seg2[3]+ (1 - model.v02_2[1,t])*A >= model.t02_2[1,t]
model.bus02_2_u2 = Constraint(model.PERIOD,rule = bus02_2_u2)

def bus02_2_t12(model,t):
    return model.t02_2[1,t] <= model.t02_2[2,t]
model.bus02_2_t12 = Constraint(model.PERIOD,rule = bus02_2_t12)

def bus02_2_t21(model,t):
    return model.t02_2[2,t]<=model.t02_2[1,t] + model.v02_2[2,t]*A
model.bus02_2_t21 = Constraint(model.PERIOD,rule = bus02_2_t21)

def bus02_2_l3(model,t):
    return bus2_seg3[0]*100*model.Pg[6,t]*model.ug[6,t] + bus2_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3[2]* model.inertia_constant[6]*model.genD_Pmax[6]*model.ug[6,t]) + bus2_seg3[3] <= model.t02_2[2,t]
model.bus02_2_l3 = Constraint(model.PERIOD,rule = bus02_2_l3)

def bus02_2_u3(model,t):
    return bus2_seg3[0]*100*model.Pg[6,t]*model.ug[6,t] + bus2_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3[2]* model.inertia_constant[6]*model.genD_Pmax[6]*model.ug[6,t]) + bus2_seg3[3] + (1 - model.v02_2[2,t])*A >= model.t02_2[2,t]
model.bus02_2_u3 = Constraint(model.PERIOD,rule = bus02_2_u3)

def bus02_2_t23(model,t):
    return model.t02_2[2,t] <= model.t02_2[3,t]
model.bus02_2_t23 = Constraint(model.PERIOD,rule = bus02_2_t23)

def bus02_2_t32(model,t):
    return model.t02_2[3,t]<=model.t02_2[2,t] + model.v02_2[3,t]*A
model.bus02_2_t32 = Constraint(model.PERIOD,rule = bus02_2_t32)

def bus02_2_l4(model,t):
    return bus2_seg4[0]*100*model.Pg[6,t]*model.ug[6,t] + bus2_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4[2]* model.inertia_constant[6]*model.genD_Pmax[6]*model.ug[6,t]) + bus2_seg4[3] <= model.t02_2[3,t]
model.bus02_2_l4 = Constraint(model.PERIOD,rule = bus02_2_l4)

def bus02_2_u4(model,t):
    return bus2_seg4[0]*100*model.Pg[6,t]*model.ug[6,t] + bus2_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4[2]* model.inertia_constant[6]*model.genD_Pmax[6]*model.ug[6,t]) + bus2_seg4[3] + (1 - model.v02_2[3,t])*A >= model.t02_2[3,t]
model.bus02_2_u4 = Constraint(model.PERIOD,rule = bus02_2_u4)

def bus02_3_R(model,t):
    return model.t02_3[3,t] <= RoCoF
model.bus02_3_R = Constraint(model.PERIOD, rule = bus02_3_R)

def bus02_3_l1(model,t):
    return bus2_seg1[0]*100*model.Pg[7,t]*model.ug[7,t] + bus2_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1[2]* model.inertia_constant[7]*model.genD_Pmax[7]*model.ug[7,t]) + bus2_seg1[3] <= model.t02_3[1,t]
model.bus02_3_l1 = Constraint(model.PERIOD, rule = bus02_3_l1)

def bus02_3_u1(model,t):
    return bus2_seg1[0]*100*model.Pg[7,t]*model.ug[7,t] + bus2_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1[2]* model.inertia_constant[7]*model.genD_Pmax[7]*model.ug[7,t]) + bus2_seg1[3] + model.v02_3[1,t]*A >= model.t02_3[1,t]
model.bus02_3_u1 = Constraint(model.PERIOD, rule = bus02_3_u1)

def bus02_3_l2(model,t):
    return bus2_seg2[0]*100*model.Pg[7,t]*model.ug[7,t] + bus2_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2[2]* model.inertia_constant[7]*model.genD_Pmax[7]*model.ug[7,t]) + bus2_seg2[3] <= model.t02_3[1,t]
model.bus02_3_l2 = Constraint(model.PERIOD,rule = bus02_3_l2)

def bus02_3_u2(model,t):
    return bus2_seg2[0]*100*model.Pg[7,t]*model.ug[7,t] + bus2_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2[2]* model.inertia_constant[7]*model.genD_Pmax[7]*model.ug[7,t]) + bus2_seg2[3]+ (1 - model.v02_3[1,t])*A >= model.t02_3[1,t]
model.bus02_3_u2 = Constraint(model.PERIOD,rule = bus02_3_u2)

def bus02_3_t12(model,t):
    return model.t02_3[1,t] <= model.t02_3[2,t]
model.bus02_3_t12 = Constraint(model.PERIOD,rule = bus02_3_t12)

def bus02_3_t21(model,t):
    return model.t02_3[2,t]<=model.t02_3[1,t] + model.v02_3[2,t]*A
model.bus02_3_t21 = Constraint(model.PERIOD,rule = bus02_3_t21)

def bus02_3_l3(model,t):
    return bus2_seg3[0]*100*model.Pg[7,t]*model.ug[7,t] + bus2_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3[2]* model.inertia_constant[7]*model.genD_Pmax[7]*model.ug[7,t]) + bus2_seg3[3]  <= model.t02_3[2,t]
model.bus02_3_l3 = Constraint(model.PERIOD,rule = bus02_3_l3)

def bus02_3_u3(model,t):
    return bus2_seg3[0]*100*model.Pg[7,t]*model.ug[7,t] + bus2_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3[2]* model.inertia_constant[7]*model.genD_Pmax[7]*model.ug[7,t]) + bus2_seg3[3] + (1 - model.v02_3[2,t])*A >= model.t02_3[2,t]
model.bus02_3_u3 = Constraint(model.PERIOD,rule = bus02_3_u3)

def bus02_3_t23(model,t):
    return model.t02_3[2,t] <= model.t02_3[3,t]
model.bus02_3_t23 = Constraint(model.PERIOD,rule = bus02_3_t23)

def bus02_3_t32(model,t):
    return model.t02_3[3,t]<=model.t02_3[2,t] + model.v02_3[3,t]*A
model.bus02_3_t32 = Constraint(model.PERIOD,rule = bus02_3_t32)

def bus02_3_l4(model,t):
    return bus2_seg4[0]*100*model.Pg[7,t]*model.ug[7,t] + bus2_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4[2]* model.inertia_constant[7]*model.genD_Pmax[7]*model.ug[7,t]) + bus2_seg4[3] <= model.t02_3[3,t]
model.bus02_3_l4 = Constraint(model.PERIOD,rule = bus02_3_l4)

def bus02_3_u4(model,t):
    return bus2_seg4[0]*100*model.Pg[7,t]*model.ug[7,t] + bus2_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4[2]* model.inertia_constant[7]*model.genD_Pmax[7]*model.ug[7,t]) + bus2_seg4[3] + (1 - model.v02_3[3,t])*A >= model.t02_3[3,t]
model.bus02_3_u4 = Constraint(model.PERIOD,rule = bus02_3_u4)

def bus02_4_R(model,t):
    return model.t02_4[3,t] <= RoCoF
model.bus02_4_R = Constraint(model.PERIOD, rule = bus02_4_R)

def bus02_4_l1(model,t):
    return bus2_seg1[0]*100*model.Pg[8,t]*model.ug[8,t] + bus2_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1[2]* model.inertia_constant[8]*model.genD_Pmax[8]*model.ug[8,t]) + bus2_seg1[3] <= model.t02_4[1,t]
model.bus02_4_l1 = Constraint(model.PERIOD, rule = bus02_4_l1)

def bus02_4_u1(model,t):
    return bus2_seg1[0]*100*model.Pg[8,t]*model.ug[8,t] + bus2_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1[2]* model.inertia_constant[8]*model.genD_Pmax[8]*model.ug[8,t]) + bus2_seg1[3] + model.v02_4[1,t]*A >= model.t02_4[1,t]
model.bus02_4_u1 = Constraint(model.PERIOD, rule = bus02_4_u1)

def bus02_4_l2(model,t):
    return bus2_seg2[0]*100*model.Pg[8,t]*model.ug[8,t] + bus2_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2[2]* model.inertia_constant[8]*model.genD_Pmax[8]*model.ug[8,t]) + bus2_seg2[3] <= model.t02_4[1,t]
model.bus02_4_l2 = Constraint(model.PERIOD,rule = bus02_4_l2)

def bus02_4_u2(model,t):
    return bus2_seg2[0]*100*model.Pg[8,t]*model.ug[8,t] + bus2_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2[2]* model.inertia_constant[8]*model.genD_Pmax[8]*model.ug[8,t]) + bus2_seg2[3] + (1 - model.v02_4[1,t])*A >= model.t02_4[1,t]
model.bus02_4_u2 = Constraint(model.PERIOD,rule = bus02_4_u2)

def bus02_4_t12(model,t):
    return model.t02_4[1,t] <= model.t02_4[2,t]
model.bus02_4_t12 = Constraint(model.PERIOD,rule = bus02_4_t12)

def bus02_4_t21(model,t):
    return model.t02_4[2,t]<=model.t02_4[1,t] + model.v02_4[2,t]*A
model.bus02_4_t21 = Constraint(model.PERIOD,rule = bus02_4_t21)

def bus02_4_l3(model,t):
    return bus2_seg3[0]*100*model.Pg[8,t]*model.ug[8,t] + bus2_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3[2]* model.inertia_constant[8]*model.genD_Pmax[8]*model.ug[8,t]) + bus2_seg3[3]  <= model.t02_4[2,t]
model.bus02_4_l3 = Constraint(model.PERIOD,rule = bus02_4_l3)

def bus02_4_u3(model,t):
    return bus2_seg3[0]*100*model.Pg[8,t]*model.ug[8,t] + bus2_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3[2]* model.inertia_constant[8]*model.genD_Pmax[8]*model.ug[8,t]) + bus2_seg3[3] + (1 - model.v02_4[2,t])*A >= model.t02_4[2,t]
model.bus02_4_u3 = Constraint(model.PERIOD,rule = bus02_4_u3)

def bus02_4_t23(model,t):
    return model.t02_4[2,t] <= model.t02_4[3,t]
model.bus02_4_t23 = Constraint(model.PERIOD,rule = bus02_4_t23)

def bus02_4_t32(model,t):
    return model.t02_4[3,t]<=model.t02_4[2,t] + model.v02_4[3,t]*A
model.bus02_4_t32 = Constraint(model.PERIOD,rule = bus02_4_t32)

def bus02_4_l4(model,t):
    return bus2_seg4[0]*100*model.Pg[8,t]*model.ug[8,t] + bus2_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4[2]* model.inertia_constant[8]*model.genD_Pmax[8]*model.ug[8,t]) + bus2_seg4[3] <= model.t02_4[3,t]
model.bus02_4_l4 = Constraint(model.PERIOD,rule = bus02_4_l4)

def bus02_4_u4(model,t):
    return bus2_seg4[0]*100*model.Pg[8,t]*model.ug[8,t] + bus2_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4[2]* model.inertia_constant[8]*model.genD_Pmax[8]*model.ug[8,t]) + bus2_seg4[3] + (1 - model.v02_4[3,t])*A >= model.t02_4[3,t]
model.bus02_4_u4 = Constraint(model.PERIOD,rule = bus02_4_u4)

def bus02_5_R(model,t):
    return model.t02_5[3,t] <= RoCoF
model.bus02_5_R = Constraint(model.PERIOD, rule = bus02_5_R)

def bus02_5_l1(model,t):
    return bus2_seg1[0]*100*model.Pg[9,t]*model.ug[9,t] + bus2_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1[2]* model.inertia_constant[9]*model.genD_Pmax[9]*model.ug[9,t]) + bus2_seg1[3] <= model.t02_5[1,t]
model.bus02_5_l1 = Constraint(model.PERIOD, rule = bus02_5_l1)

def bus02_5_u1(model,t):
    return bus2_seg1[0]*100*model.Pg[9,t]*model.ug[9,t] + bus2_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1[2]* model.inertia_constant[9]*model.genD_Pmax[9]*model.ug[9,t]) + bus2_seg1[3] + model.v02_5[1,t]*A >= model.t02_5[1,t]
model.bus02_5_u1 = Constraint(model.PERIOD, rule = bus02_5_u1)

def bus02_5_l2(model,t):
    return bus2_seg2[0]*100*model.Pg[9,t]*model.ug[9,t] + bus2_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2[2]* model.inertia_constant[9]*model.genD_Pmax[9]*model.ug[9,t]) + bus2_seg2[3] <= model.t02_5[1,t]
model.bus02_5_l2 = Constraint(model.PERIOD,rule = bus02_5_l2)

def bus02_5_u2(model,t):
    return bus2_seg2[0]*100*model.Pg[9,t]*model.ug[9,t] + bus2_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2[2]* model.inertia_constant[9]*model.genD_Pmax[9]*model.ug[9,t]) + bus2_seg2[3] + (1 - model.v02_5[1,t])*A >= model.t02_5[1,t]
model.bus02_5_u2 = Constraint(model.PERIOD,rule = bus02_5_u2)

def bus02_5_t12(model,t):
    return model.t02_5[1,t] <= model.t02_5[2,t]
model.bus02_5_t12 = Constraint(model.PERIOD,rule = bus02_5_t12)

def bus02_5_t21(model,t):
    return model.t02_5[2,t]<=model.t02_5[1,t] + model.v02_5[2,t]*A
model.bus02_5_t21 = Constraint(model.PERIOD,rule = bus02_5_t21)

def bus02_5_l3(model,t):
    return bus2_seg3[0]*100*model.Pg[9,t]*model.ug[9,t] + bus2_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3[2]* model.inertia_constant[9]*model.genD_Pmax[9]*model.ug[9,t]) + bus2_seg3[3]  <= model.t02_5[2,t]
model.bus02_5_l3 = Constraint(model.PERIOD,rule = bus02_5_l3)

def bus02_5_u3(model,t):
    return bus2_seg3[0]*100*model.Pg[9,t]*model.ug[9,t] + bus2_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3[2]* model.inertia_constant[9]*model.genD_Pmax[9]*model.ug[9,t]) + bus2_seg3[3] + (1 - model.v02_5[2,t])*A >= model.t02_5[2,t]
model.bus02_5_u3 = Constraint(model.PERIOD,rule = bus02_5_u3)

def bus02_5_t23(model,t):
    return model.t02_5[2,t] <= model.t02_5[3,t]
model.bus02_5_t23 = Constraint(model.PERIOD,rule = bus02_5_t23)

def bus02_5_t32(model,t):
    return model.t02_5[3,t]<=model.t02_5[2,t] + model.v02_5[3,t]*A
model.bus02_5_t32 = Constraint(model.PERIOD,rule = bus02_5_t32)

def bus02_5_l4(model,t):
    return bus2_seg4[0]*100*model.Pg[9,t]*model.ug[9,t] + bus2_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4[2]* model.inertia_constant[9]*model.genD_Pmax[9]*model.ug[9,t]) + bus2_seg4[3] <= model.t02_5[3,t]
model.bus02_5_l4 = Constraint(model.PERIOD,rule = bus02_5_l4)

def bus02_5_u4(model,t):
    return bus2_seg4[0]*100*model.Pg[9,t]*model.ug[9,t] + bus2_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4[2]* model.inertia_constant[9]*model.genD_Pmax[9]*model.ug[9,t]) + bus2_seg4[3] + (1 - model.v02_5[3,t])*A >= model.t02_5[3,t]
model.bus02_5_u4 = Constraint(model.PERIOD,rule = bus02_5_u4)

def bus02_6_R(model,t):
    return model.t02_6[3,t] <= RoCoF
model.bus02_6_R = Constraint(model.PERIOD, rule = bus02_6_R)

def bus02_6_l1(model,t):
    return bus2_seg1[0]*100*model.Pg[10,t]*model.ug[10,t] + bus2_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1[2]* model.inertia_constant[10]*model.genD_Pmax[10]*model.ug[10,t]) + bus2_seg1[3] <= model.t02_6[1,t]
model.bus02_6_l1 = Constraint(model.PERIOD, rule = bus02_6_l1)

def bus02_6_u1(model,t):
    return bus2_seg1[0]*100*model.Pg[10,t]*model.ug[10,t] + bus2_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1[2]* model.inertia_constant[10]*model.genD_Pmax[10]*model.ug[10,t]) + bus2_seg1[3] + model.v02_6[1,t]*A >= model.t02_6[1,t]
model.bus02_6_u1 = Constraint(model.PERIOD, rule = bus02_6_u1)

def bus02_6_l2(model,t):
    return bus2_seg2[0]*100*model.Pg[10,t]*model.ug[10,t] + bus2_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2[2]* model.inertia_constant[10]*model.genD_Pmax[10]*model.ug[10,t]) + bus2_seg2[3] <= model.t02_6[1,t]
model.bus02_6_l2 = Constraint(model.PERIOD,rule = bus02_6_l2)

def bus02_6_u2(model,t):
    return bus2_seg2[0]*100*model.Pg[10,t]*model.ug[10,t] + bus2_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2[2]* model.inertia_constant[10]*model.genD_Pmax[10]*model.ug[10,t]) + bus2_seg2[3] + (1 - model.v02_6[1,t])*A >= model.t02_6[1,t]
model.bus02_6_u2 = Constraint(model.PERIOD,rule = bus02_6_u2)

def bus02_6_t12(model,t):
    return model.t02_6[1,t] <= model.t02_6[2,t]
model.bus02_6_t12 = Constraint(model.PERIOD,rule = bus02_6_t12)

def bus02_6_t21(model,t):
    return model.t02_6[2,t]<=model.t02_6[1,t] + model.v02_6[2,t]*A
model.bus02_6_t21 = Constraint(model.PERIOD,rule = bus02_6_t21)

def bus02_6_l3(model,t):
    return bus2_seg3[0]*100*model.Pg[10,t]*model.ug[10,t] + bus2_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3[2]* model.inertia_constant[10]*model.genD_Pmax[10]*model.ug[10,t]) + bus2_seg3[3]  <= model.t02_6[2,t]
model.bus02_6_l3 = Constraint(model.PERIOD,rule = bus02_6_l3)

def bus02_6_u3(model,t):
    return bus2_seg3[0]*100*model.Pg[10,t]*model.ug[10,t] + bus2_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3[2]* model.inertia_constant[10]*model.genD_Pmax[10]*model.ug[10,t]) + bus2_seg3[3] + (1 - model.v02_6[2,t])*A >= model.t02_6[2,t]
model.bus02_6_u3 = Constraint(model.PERIOD,rule = bus02_6_u3)

def bus02_6_t23(model,t):
    return model.t02_6[2,t] <= model.t02_6[3,t]
model.bus02_6_t23 = Constraint(model.PERIOD,rule = bus02_6_t23)

def bus02_6_t32(model,t):
    return model.t02_6[3,t]<=model.t02_6[2,t] + model.v02_6[3,t]*A
model.bus02_6_t32 = Constraint(model.PERIOD,rule = bus02_6_t32)

def bus02_6_l4(model,t):
    return bus2_seg4[0]*100*model.Pg[10,t]*model.ug[10,t] + bus2_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4[2]* model.inertia_constant[10]*model.genD_Pmax[10]*model.ug[10,t]) + bus2_seg4[3] <= model.t02_6[3,t]
model.bus02_6_l4 = Constraint(model.PERIOD,rule = bus02_6_l4)

def bus02_6_u4(model,t):
    return bus2_seg4[0]*100*model.Pg[10,t]*model.ug[10,t] + bus2_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4[2]* model.inertia_constant[10]*model.genD_Pmax[10]*model.ug[10,t]) + bus2_seg4[3] + (1 - model.v02_6[3,t])*A >= model.t02_6[3,t]
model.bus02_6_u4 = Constraint(model.PERIOD,rule = bus02_6_u4)

# Node 07 locational RoCoF constraints

def bus07_1_R(model,t):
    return model.t07_1[3,t] <= RoCoF
model.bus07_1_R = Constraint(model.PERIOD, rule = bus07_1_R)

def bus07_1_l1(model,t):
    return bus7_seg1[0]*100*model.Pg[11,t]*model.ug[11,t] + bus7_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg1[2]* model.inertia_constant[11]*model.genD_Pmax[11]*model.ug[11,t]) + bus7_seg1[3]<= model.t07_1[1,t]
model.bus07_1_l1 = Constraint(model.PERIOD, rule = bus07_1_l1)

def bus07_1_u1(model,t):
    return bus7_seg1[0]*100*model.Pg[11,t]*model.ug[11,t] + bus7_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg1[2]* model.inertia_constant[11]*model.genD_Pmax[11]*model.ug[11,t]) + bus7_seg1[3] + model.v07_1[1,t]*A >= model.t07_1[1,t]
model.bus07_1_u1 = Constraint(model.PERIOD, rule = bus07_1_u1)

def bus07_1_l2(model,t):
    return bus7_seg2[0]*100*model.Pg[11,t]*model.ug[11,t] + bus7_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg2[2]* model.inertia_constant[11]*model.genD_Pmax[11]*model.ug[11,t]) + bus7_seg2[3] <= model.t07_1[1,t]
model.bus07_1_l2 = Constraint(model.PERIOD,rule = bus07_1_l2)

def bus07_1_u2(model,t):
    return bus7_seg2[0]*100*model.Pg[11,t]*model.ug[11,t] + bus7_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg2[2]* model.inertia_constant[11]*model.genD_Pmax[11]*model.ug[11,t]) + bus7_seg2[3] + (1 - model.v07_1[1,t])*A >= model.t07_1[1,t]
model.bus07_1_u2 = Constraint(model.PERIOD,rule = bus07_1_u2)

def bus07_1_t12(model,t):
    return model.t07_1[1,t] <= model.t07_1[2,t]
model.bus07_1_t12 = Constraint(model.PERIOD,rule = bus07_1_t12)

def bus07_1_t21(model,t):
    return model.t07_1[2,t]<=model.t07_1[1,t] + model.v07_1[2,t]*A
model.bus07_1_t21 = Constraint(model.PERIOD,rule = bus07_1_t21)

def bus07_1_l3(model,t):
    return bus7_seg3[0]*100*model.Pg[11,t]*model.ug[11,t] + bus7_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg3[2]* model.inertia_constant[11]*model.genD_Pmax[11]*model.ug[11,t]) + bus7_seg3[3] <= model.t07_1[2,t]
model.bus07_1_l3 = Constraint(model.PERIOD,rule = bus07_1_l3)

def bus07_1_u3(model,t):
    return bus7_seg3[0]*100*model.Pg[11,t]*model.ug[11,t] + bus7_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg3[2]* model.inertia_constant[11]*model.genD_Pmax[11]*model.ug[11,t]) + bus7_seg3[3]+ (1 - model.v07_1[2,t])*A >= model.t07_1[2,t]
model.bus07_1_u3 = Constraint(model.PERIOD,rule = bus07_1_u3)

def bus07_1_t23(model,t):
    return model.t07_1[2,t] <= model.t07_1[3,t]
model.bus07_1_t23 = Constraint(model.PERIOD,rule = bus07_1_t23)

def bus07_1_t32(model,t):
    return model.t07_1[3,t]<=model.t07_1[2,t] + model.v07_1[3,t]*A
model.bus07_1_t32 = Constraint(model.PERIOD,rule = bus07_1_t32)

def bus07_1_l4(model,t):
    return bus7_seg4[0]*100*model.Pg[11,t]*model.ug[11,t] + bus7_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg4[2]* model.inertia_constant[11]*model.genD_Pmax[11]*model.ug[11,t]) + bus7_seg4[3] <= model.t07_1[3,t]
model.bus07_1_l4 = Constraint(model.PERIOD,rule = bus07_1_l4)

def bus07_1_u4(model,t):
    return bus7_seg4[0]*100*model.Pg[11,t]*model.ug[11,t] + bus7_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg4[2]* model.inertia_constant[11]*model.genD_Pmax[11]*model.ug[11,t]) + bus7_seg4[3] + (1 - model.v07_1[3,t])*A >= model.t07_1[3,t]
model.bus07_1_u4 = Constraint(model.PERIOD,rule = bus07_1_u4)

def bus07_2_R(model,t):
    return model.t07_2[3,t] <= RoCoF
model.bus07_2_R = Constraint(model.PERIOD, rule = bus07_2_R)

def bus07_2_l1(model,t):
    return bus7_seg1[0]*100*model.Pg[12,t]*model.ug[12,t] + bus7_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg1[2]* model.inertia_constant[12]*model.genD_Pmax[12]*model.ug[12,t]) + bus7_seg1[3] <= model.t07_2[1,t]
model.bus07_2_l1 = Constraint(model.PERIOD, rule = bus07_2_l1)

def bus07_2_u1(model,t):
    return bus7_seg1[0]*100*model.Pg[12,t]*model.ug[12,t] + bus7_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg1[2]* model.inertia_constant[12]*model.genD_Pmax[12]*model.ug[12,t]) + bus7_seg1[3]+ model.v07_2[1,t]*A >= model.t07_2[1,t]
model.bus07_2_u1 = Constraint(model.PERIOD, rule = bus07_2_u1)

def bus07_2_l2(model,t):
    return bus7_seg2[0]*100*model.Pg[12,t]*model.ug[12,t] + bus7_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg2[2]* model.inertia_constant[12]*model.genD_Pmax[12]*model.ug[12,t]) + bus7_seg2[3] <= model.t07_2[1,t]
model.bus07_2_l2 = Constraint(model.PERIOD,rule = bus07_2_l2)

def bus07_2_u2(model,t):
    return bus7_seg2[0]*100*model.Pg[12,t]*model.ug[12,t] + bus7_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg2[2]* model.inertia_constant[12]*model.genD_Pmax[12]*model.ug[12,t]) + bus7_seg2[3] + (1 - model.v07_2[1,t])*A >= model.t07_2[1,t]
model.bus07_2_u2 = Constraint(model.PERIOD,rule = bus07_2_u2)

def bus07_2_t12(model,t):
    return model.t07_2[1,t] <= model.t07_2[2,t]
model.bus07_2_t12 = Constraint(model.PERIOD,rule = bus07_2_t12)

def bus07_2_t21(model,t):
    return model.t07_2[2,t]<=model.t07_2[1,t] + model.v07_2[2,t]*A
model.bus07_2_t21 = Constraint(model.PERIOD,rule = bus07_2_t21)

def bus07_2_l3(model,t):
    return bus7_seg3[0]*100*model.Pg[12,t]*model.ug[12,t] + bus7_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg3[2]* model.inertia_constant[12]*model.genD_Pmax[12]*model.ug[12,t]) + bus7_seg3[3] <= model.t07_2[2,t]
model.bus07_2_l3 = Constraint(model.PERIOD,rule = bus07_2_l3)

def bus07_2_u3(model,t):
    return bus7_seg3[0]*100*model.Pg[12,t]*model.ug[12,t] + bus7_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg3[2]* model.inertia_constant[12]*model.genD_Pmax[12]*model.ug[12,t]) + bus7_seg3[3] + (1 - model.v07_2[2,t])*A >= model.t07_2[2,t]
model.bus07_2_u3 = Constraint(model.PERIOD,rule = bus07_2_u3)

def bus07_2_t23(model,t):
    return model.t07_2[2,t] <= model.t07_2[3,t]
model.bus07_2_t23 = Constraint(model.PERIOD,rule = bus07_2_t23)

def bus07_2_t32(model,t):
    return model.t07_2[3,t]<=model.t07_2[2,t] + model.v07_2[3,t]*A
model.bus07_2_t32 = Constraint(model.PERIOD,rule = bus07_2_t32)

def bus07_2_l4(model,t):
    return bus7_seg4[0]*100*model.Pg[12,t]*model.ug[12,t] + bus7_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg4[2]* model.inertia_constant[12]*model.genD_Pmax[12]*model.ug[12,t]) + bus7_seg4[3] <= model.t07_2[3,t]
model.bus07_2_l4 = Constraint(model.PERIOD,rule = bus07_2_l4)

def bus07_2_u4(model,t):
    return bus7_seg4[0]*100*model.Pg[12,t]*model.ug[12,t] + bus7_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg4[2]* model.inertia_constant[12]*model.genD_Pmax[12]*model.ug[12,t]) + bus7_seg4[3] + (1 - model.v07_2[3,t])*A >= model.t07_2[3,t]
model.bus07_2_u4 = Constraint(model.PERIOD,rule = bus07_2_u4)

def bus07_3_R(model,t):
    return model.t07_3[3,t] <= RoCoF
model.bus07_3_R = Constraint(model.PERIOD, rule = bus07_3_R)

def bus07_3_l1(model,t):
    return bus7_seg1[0]*100*model.Pg[13,t]*model.ug[13,t] + bus7_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg1[2]* model.inertia_constant[13]*model.genD_Pmax[13]*model.ug[13,t]) + bus7_seg1[3] <= model.t07_3[1,t]
model.bus07_3_l1 = Constraint(model.PERIOD, rule = bus07_3_l1)

def bus07_3_u1(model,t):
    return bus7_seg1[0]*100*model.Pg[13,t]*model.ug[13,t] + bus7_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg1[2]* model.inertia_constant[13]*model.genD_Pmax[13]*model.ug[13,t]) + bus7_seg1[3] + model.v07_3[1,t]*A >= model.t07_3[1,t]
model.bus07_3_u1 = Constraint(model.PERIOD, rule = bus07_3_u1)

def bus07_3_l2(model,t):
    return bus7_seg2[0]*100*model.Pg[13,t]*model.ug[13,t] + bus7_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg2[2]* model.inertia_constant[13]*model.genD_Pmax[13]*model.ug[13,t]) + bus7_seg2[3] <= model.t07_3[1,t]
model.bus07_3_l2 = Constraint(model.PERIOD,rule = bus07_3_l2)

def bus07_3_u2(model,t):
    return bus7_seg2[0]*100*model.Pg[13,t]*model.ug[13,t] + bus7_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg2[2]* model.inertia_constant[13]*model.genD_Pmax[13]*model.ug[13,t]) + bus7_seg2[3] + (1 - model.v07_3[1,t])*A >= model.t07_3[1,t]
model.bus07_3_u2 = Constraint(model.PERIOD,rule = bus07_3_u2)

def bus07_3_t12(model,t):
    return model.t07_3[1,t] <= model.t07_3[2,t]
model.bus07_3_t12 = Constraint(model.PERIOD,rule = bus07_3_t12)

def bus07_3_t21(model,t):
    return model.t07_3[2,t]<=model.t07_3[1,t] + model.v07_3[2,t]*A
model.bus07_3_t21 = Constraint(model.PERIOD,rule = bus07_3_t21)

def bus07_3_l3(model,t):
    return bus7_seg3[0]*100*model.Pg[13,t]*model.ug[13,t] + bus7_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg3[2]* model.inertia_constant[13]*model.genD_Pmax[13]*model.ug[13,t]) + bus7_seg3[3]  <= model.t07_3[2,t]
model.bus07_3_l3 = Constraint(model.PERIOD,rule = bus07_3_l3)

def bus07_3_u3(model,t):
    return bus7_seg3[0]*100*model.Pg[13,t]*model.ug[13,t] + bus7_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg3[2]* model.inertia_constant[13]*model.genD_Pmax[13]*model.ug[13,t]) + bus7_seg3[3]  + (1 - model.v07_3[2,t])*A >= model.t07_3[2,t]
model.bus07_3_u3 = Constraint(model.PERIOD,rule = bus07_3_u3)

def bus07_3_t23(model,t):
    return model.t07_3[2,t] <= model.t07_3[3,t]
model.bus07_3_t23 = Constraint(model.PERIOD,rule = bus07_3_t23)

def bus07_3_t32(model,t):
    return model.t07_3[3,t]<=model.t07_3[2,t] + model.v07_3[3,t]*A
model.bus07_3_t32 = Constraint(model.PERIOD,rule = bus07_3_t32)

def bus07_3_l4(model,t):
    return bus7_seg4[0]*100*model.Pg[13,t]*model.ug[13,t] + bus7_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg4[2]* model.inertia_constant[13]*model.genD_Pmax[13]*model.ug[13,t]) + bus7_seg4[3] <= model.t07_3[3,t]
model.bus07_3_l4 = Constraint(model.PERIOD,rule = bus07_3_l4)

def bus07_3_u4(model,t):
    return bus7_seg4[0]*100*model.Pg[13,t]*model.ug[13,t] + bus7_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg4[2]* model.inertia_constant[13]*model.genD_Pmax[13]*model.ug[13,t]) + bus7_seg4[3] + (1 - model.v07_3[3,t])*A >= model.t07_3[3,t]
model.bus07_3_u4 = Constraint(model.PERIOD,rule = bus07_3_u4)

# Node 13 locational RoCoF constraints

def bus13_1_R(model,t):
    return model.t13_1[3,t] <= RoCoF
model.bus13_1_R = Constraint(model.PERIOD, rule = bus13_1_R)

def bus13_1_l1(model,t):
    return bus13_seg1[0]*100*model.Pg[14,t]*model.ug[14,t] + bus13_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1[2]* model.inertia_constant[14]*model.genD_Pmax[14]*model.ug[14,t]) + bus13_seg1[3] <= model.t13_1[1,t]
model.bus13_1_l1 = Constraint(model.PERIOD, rule = bus13_1_l1)

def bus13_1_u1(model,t):
    return bus13_seg1[0]*100*model.Pg[14,t]*model.ug[14,t] + bus13_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1[2]* model.inertia_constant[14]*model.genD_Pmax[14]*model.ug[14,t]) + bus13_seg1[3] + model.v13_1[1,t]*A >= model.t13_1[1,t]
model.bus13_1_u1 = Constraint(model.PERIOD, rule = bus13_1_u1)

def bus13_1_l2(model,t):
    return bus13_seg2[0]*100*model.Pg[14,t]*model.ug[14,t] + bus13_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2[2]* model.inertia_constant[14]*model.genD_Pmax[14]*model.ug[14,t]) + bus13_seg2[3] <= model.t13_1[1,t]
model.bus13_1_l2 = Constraint(model.PERIOD,rule = bus13_1_l2)

def bus13_1_u2(model,t):
    return bus13_seg2[0]*100*model.Pg[14,t]*model.ug[14,t] + bus13_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2[2]* model.inertia_constant[14]*model.genD_Pmax[14]*model.ug[14,t]) + bus13_seg2[3] + (1 - model.v13_1[1,t])*A >= model.t13_1[1,t]
model.bus13_1_u2 = Constraint(model.PERIOD,rule = bus13_1_u2)

def bus13_1_t12(model,t):
    return model.t13_1[1,t] <= model.t13_1[2,t]
model.bus13_1_t12 = Constraint(model.PERIOD,rule = bus13_1_t12)

def bus13_1_t21(model,t):
    return model.t13_1[2,t]<=model.t13_1[1,t] + model.v13_1[2,t]*A
model.bus13_1_t21 = Constraint(model.PERIOD,rule = bus13_1_t21)

def bus13_1_l3(model,t):
    return bus13_seg3[0]*100*model.Pg[14,t]*model.ug[14,t] + bus13_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3[2]* model.inertia_constant[14]*model.genD_Pmax[14]*model.ug[14,t]) + bus13_seg3[3] <= model.t13_1[2,t]
model.bus13_1_l3 = Constraint(model.PERIOD,rule = bus13_1_l3)

def bus13_1_u3(model,t):
    return bus13_seg3[0]*100*model.Pg[14,t]*model.ug[14,t] + bus13_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3[2]* model.inertia_constant[14]*model.genD_Pmax[14]*model.ug[14,t]) + bus13_seg3[3] + (1 - model.v13_1[2,t])*A >= model.t13_1[2,t]
model.bus13_1_u3 = Constraint(model.PERIOD,rule = bus13_1_u3)

def bus13_1_t23(model,t):
    return model.t13_1[2,t] <= model.t13_1[3,t]
model.bus13_1_t23 = Constraint(model.PERIOD,rule = bus13_1_t23)

def bus13_1_t32(model,t):
    return model.t13_1[3,t]<=model.t13_1[2,t] + model.v13_1[3,t]*A
model.bus13_1_t32 = Constraint(model.PERIOD,rule = bus13_1_t32)

def bus13_1_l4(model,t):
    return bus13_seg4[0]*100*model.Pg[14,t]*model.ug[14,t] + bus13_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4[2]* model.inertia_constant[14]*model.genD_Pmax[14]*model.ug[14,t]) + bus13_seg4[3] <= model.t13_1[3,t]
model.bus13_1_l4 = Constraint(model.PERIOD,rule = bus13_1_l4)

def bus13_1_u4(model,t):
    return bus13_seg4[0]*100*model.Pg[14,t]*model.ug[14,t] + bus13_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4[2]* model.inertia_constant[14]*model.genD_Pmax[14]*model.ug[14,t]) + bus13_seg4[3] + (1 - model.v13_1[3,t])*A >= model.t13_1[3,t]
model.bus13_1_u4 = Constraint(model.PERIOD,rule = bus13_1_u4)

def bus13_2_R(model,t):
    return model.t13_2[3,t] <= RoCoF
model.bus13_2_R = Constraint(model.PERIOD, rule = bus13_2_R)

def bus13_2_l1(model,t):
    return bus13_seg1[0]*100*model.Pg[15,t]*model.ug[15,t] + bus13_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1[2]* model.inertia_constant[15]*model.genD_Pmax[15]*model.ug[15,t]) + bus13_seg1[3] <= model.t13_2[1,t]
model.bus13_2_l1 = Constraint(model.PERIOD, rule = bus13_2_l1)

def bus13_2_u1(model,t):
    return bus13_seg1[0]*100*model.Pg[15,t]*model.ug[15,t] + bus13_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1[2]* model.inertia_constant[15]*model.genD_Pmax[15]*model.ug[15,t]) + bus13_seg1[3] + model.v13_2[1,t]*A >= model.t13_2[1,t]
model.bus13_2_u1 = Constraint(model.PERIOD, rule = bus13_2_u1)

def bus13_2_l2(model,t):
    return bus13_seg2[0]*100*model.Pg[15,t]*model.ug[15,t] + bus13_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2[2]* model.inertia_constant[15]*model.genD_Pmax[15]*model.ug[15,t]) + bus13_seg2[3] <= model.t13_2[1,t]
model.bus13_2_l2 = Constraint(model.PERIOD,rule = bus13_2_l2)

def bus13_2_u2(model,t):
    return bus13_seg2[0]*100*model.Pg[15,t]*model.ug[15,t] + bus13_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2[2]* model.inertia_constant[15]*model.genD_Pmax[15]*model.ug[15,t]) + bus13_seg2[3]+ (1 - model.v13_2[1,t])*A >= model.t13_2[1,t]
model.bus13_2_u2 = Constraint(model.PERIOD,rule = bus13_2_u2)

def bus13_2_t12(model,t):
    return model.t13_2[1,t] <= model.t13_2[2,t]
model.bus13_2_t12 = Constraint(model.PERIOD,rule = bus13_2_t12)

def bus13_2_t21(model,t):
    return model.t13_2[2,t]<=model.t13_2[1,t] + model.v13_2[2,t]*A
model.bus13_2_t21 = Constraint(model.PERIOD,rule = bus13_2_t21)

def bus13_2_l3(model,t):
    return bus13_seg3[0]*100*model.Pg[15,t]*model.ug[15,t] + bus13_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3[2]* model.inertia_constant[15]*model.genD_Pmax[15]*model.ug[15,t]) + bus13_seg3[3]  <= model.t13_2[2,t]
model.bus13_2_l3 = Constraint(model.PERIOD,rule = bus13_2_l3)

def bus13_2_u3(model,t):
    return bus13_seg3[0]*100*model.Pg[15,t]*model.ug[15,t] + bus13_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3[2]* model.inertia_constant[15]*model.genD_Pmax[15]*model.ug[15,t]) + bus13_seg3[3] + (1 - model.v13_2[2,t])*A >= model.t13_2[2,t]
model.bus13_2_u3 = Constraint(model.PERIOD,rule = bus13_2_u3)

def bus13_2_t23(model,t):
    return model.t13_2[2,t] <= model.t13_2[3,t]
model.bus13_2_t23 = Constraint(model.PERIOD,rule = bus13_2_t23)

def bus13_2_t32(model,t):
    return model.t13_2[3,t]<=model.t13_2[2,t] + model.v13_2[3,t]*A
model.bus13_2_t32 = Constraint(model.PERIOD,rule = bus13_2_t32)

def bus13_2_l4(model,t):
    return bus13_seg4[0]*100*model.Pg[15,t]*model.ug[15,t] + bus13_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4[2]* model.inertia_constant[15]*model.genD_Pmax[15]*model.ug[15,t]) + bus13_seg4[3] <= model.t13_2[3,t]
model.bus13_2_l4 = Constraint(model.PERIOD,rule = bus13_2_l4)

def bus13_2_u4(model,t):
    return bus13_seg4[0]*100*model.Pg[15,t]*model.ug[15,t] + bus13_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4[2]* model.inertia_constant[15]*model.genD_Pmax[15]*model.ug[15,t]) + bus13_seg4[3] + (1 - model.v13_2[3,t])*A >= model.t13_2[3,t]
model.bus13_2_u4 = Constraint(model.PERIOD,rule = bus13_2_u4)

def bus13_3_R(model,t):
    return model.t13_3[3,t] <= RoCoF
model.bus13_3_R = Constraint(model.PERIOD, rule = bus13_3_R)

def bus13_3_l1(model,t):
    return bus13_seg1[0]*100*model.Pg[16,t]*model.ug[16,t] + bus13_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1[2]* model.inertia_constant[16]*model.genD_Pmax[16]*model.ug[16,t]) + bus13_seg1[3] <= model.t13_3[1,t]
model.bus13_3_l1 = Constraint(model.PERIOD, rule = bus13_3_l1)

def bus13_3_u1(model,t):
    return bus13_seg1[0]*100*model.Pg[16,t]*model.ug[16,t] + bus13_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1[2]* model.inertia_constant[16]*model.genD_Pmax[16]*model.ug[16,t]) + bus13_seg1[3] + model.v13_3[1,t]*A >= model.t13_3[1,t]
model.bus13_3_u1 = Constraint(model.PERIOD, rule = bus13_3_u1)

def bus13_3_l2(model,t):
    return bus13_seg2[0]*100*model.Pg[16,t]*model.ug[16,t] + bus13_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2[2]* model.inertia_constant[16]*model.genD_Pmax[16]*model.ug[16,t]) + bus13_seg2[3] <= model.t13_3[1,t]
model.bus13_3_l2 = Constraint(model.PERIOD,rule = bus13_3_l2)

def bus13_3_u2(model,t):
    return bus13_seg2[0]*100*model.Pg[16,t]*model.ug[16,t] + bus13_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2[2]* model.inertia_constant[16]*model.genD_Pmax[16]*model.ug[16,t]) + bus13_seg2[3] + (1 - model.v13_3[1,t])*A >= model.t13_3[1,t]
model.bus13_3_u2 = Constraint(model.PERIOD,rule = bus13_3_u2)

def bus13_3_t12(model,t):
    return model.t13_3[1,t] <= model.t13_3[2,t]
model.bus13_3_t12 = Constraint(model.PERIOD,rule = bus13_3_t12)

def bus13_3_t21(model,t):
    return model.t13_3[2,t]<=model.t13_3[1,t] + model.v13_3[2,t]*A
model.bus13_3_t21 = Constraint(model.PERIOD,rule = bus13_3_t21)

def bus13_3_l3(model,t):
    return bus13_seg3[0]*100*model.Pg[16,t]*model.ug[16,t] + bus13_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3[2]* model.inertia_constant[16]*model.genD_Pmax[16]*model.ug[16,t]) + bus13_seg3[3] <= model.t13_3[2,t]
model.bus13_3_l3 = Constraint(model.PERIOD,rule = bus13_3_l3)

def bus13_3_u3(model,t):
    return bus13_seg3[0]*100*model.Pg[16,t]*model.ug[16,t] + bus13_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3[2]* model.inertia_constant[16]*model.genD_Pmax[16]*model.ug[16,t]) + bus13_seg3[3] + (1 - model.v13_3[2,t])*A >= model.t13_3[2,t]
model.bus13_3_u3 = Constraint(model.PERIOD,rule = bus13_3_u3)

def bus13_3_t23(model,t):
    return model.t13_3[2,t] <= model.t13_3[3,t]
model.bus13_3_t23 = Constraint(model.PERIOD,rule = bus13_3_t23)

def bus13_3_t32(model,t):
    return model.t13_3[3,t]<=model.t13_3[2,t] + model.v13_3[3,t]*A
model.bus13_3_t32 = Constraint(model.PERIOD,rule = bus13_3_t32)

def bus13_3_l4(model,t):
    return bus13_seg4[0]*100*model.Pg[16,t]*model.ug[16,t] + bus13_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4[2]* model.inertia_constant[16]*model.genD_Pmax[16]*model.ug[16,t]) + bus13_seg4[3] <= model.t13_3[3,t]
model.bus13_3_l4 = Constraint(model.PERIOD,rule = bus13_3_l4)

def bus13_3_u4(model,t):
    return bus13_seg4[0]*100*model.Pg[16,t]*model.ug[16,t] + bus13_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4[2]* model.inertia_constant[16]*model.genD_Pmax[16]*model.ug[16,t]) + bus13_seg4[3] + (1 - model.v13_3[3,t])*A >= model.t13_3[3,t]
model.bus13_3_u4 = Constraint(model.PERIOD,rule = bus13_3_u4)

def bus13_4_R(model,t):
    return model.t13_4[3,t] <= RoCoF
model.bus13_4_R = Constraint(model.PERIOD, rule = bus13_4_R)

def bus13_4_l1(model,t):
    return bus13_seg1[0]*100*model.Pg[17,t]*model.ug[17,t] + bus13_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1[2]* model.inertia_constant[17]*model.genD_Pmax[17]*model.ug[17,t]) + bus13_seg1[3] <= model.t13_4[1,t]
model.bus13_4_l1 = Constraint(model.PERIOD, rule = bus13_4_l1)

def bus13_4_u1(model,t):
    return bus13_seg1[0]*100*model.Pg[17,t]*model.ug[17,t] + bus13_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1[2]* model.inertia_constant[17]*model.genD_Pmax[17]*model.ug[17,t]) + bus13_seg1[3] + model.v13_4[1,t]*A >= model.t13_4[1,t]
model.bus13_4_u1 = Constraint(model.PERIOD, rule = bus13_4_u1)

def bus13_4_l2(model,t):
    return bus13_seg2[0]*100*model.Pg[17,t]*model.ug[17,t] + bus13_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2[2]* model.inertia_constant[17]*model.genD_Pmax[17]*model.ug[17,t]) + bus13_seg2[3] <= model.t13_4[1,t]
model.bus13_4_l2 = Constraint(model.PERIOD,rule = bus13_4_l2)

def bus13_4_u2(model,t):
    return bus13_seg2[0]*100*model.Pg[17,t]*model.ug[17,t] + bus13_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2[2]* model.inertia_constant[17]*model.genD_Pmax[17]*model.ug[17,t]) + bus13_seg2[3]+ (1 - model.v13_4[1,t])*A >= model.t13_4[1,t]
model.bus13_4_u2 = Constraint(model.PERIOD,rule = bus13_4_u2)

def bus13_4_t12(model,t):
    return model.t13_4[1,t] <= model.t13_4[2,t]
model.bus13_4_t12 = Constraint(model.PERIOD,rule = bus13_4_t12)

def bus13_4_t21(model,t):
    return model.t13_4[2,t]<=model.t13_4[1,t] + model.v13_4[2,t]*A
model.bus13_4_t21 = Constraint(model.PERIOD,rule = bus13_4_t21)

def bus13_4_l3(model,t):
    return bus13_seg3[0]*100*model.Pg[17,t]*model.ug[17,t] + bus13_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3[2]* model.inertia_constant[17]*model.genD_Pmax[17]*model.ug[17,t]) + bus13_seg3[3]  <= model.t13_4[2,t]
model.bus13_4_l3 = Constraint(model.PERIOD,rule = bus13_4_l3)

def bus13_4_u3(model,t):
    return bus13_seg3[0]*100*model.Pg[17,t]*model.ug[17,t] + bus13_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3[2]* model.inertia_constant[17]*model.genD_Pmax[17]*model.ug[17,t]) + bus13_seg3[3] + (1 - model.v13_4[2,t])*A >= model.t13_4[2,t]
model.bus13_4_u3 = Constraint(model.PERIOD,rule = bus13_4_u3)

def bus13_4_t23(model,t):
    return model.t13_4[2,t] <= model.t13_4[3,t]
model.bus13_4_t23 = Constraint(model.PERIOD,rule = bus13_4_t23)

def bus13_4_t32(model,t):
    return model.t13_4[3,t]<=model.t13_4[2,t] + model.v13_4[3,t]*A
model.bus13_4_t32 = Constraint(model.PERIOD,rule = bus13_4_t32)

def bus13_4_l4(model,t):
    return bus13_seg4[0]*100*model.Pg[17,t]*model.ug[17,t] + bus13_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4[2]* model.inertia_constant[17]*model.genD_Pmax[17]*model.ug[17,t]) + bus13_seg4[3] <= model.t13_4[3,t]
model.bus13_4_l4 = Constraint(model.PERIOD,rule = bus13_4_l4)

def bus13_4_u4(model,t):
    return bus13_seg4[0]*100*model.Pg[17,t]*model.ug[17,t] + bus13_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4[2]* model.inertia_constant[17]*model.genD_Pmax[17]*model.ug[17,t]) + bus13_seg4[3] + (1 - model.v13_4[3,t])*A >= model.t13_4[3,t]
model.bus13_4_u4 = Constraint(model.PERIOD,rule = bus13_4_u4)

def bus13_5_R(model,t):
    return model.t13_5[3,t] <= RoCoF
model.bus13_5_R = Constraint(model.PERIOD, rule = bus13_5_R)

def bus13_5_l1(model,t):
    return bus13_seg1[0]*100*model.Pg[18,t]*model.ug[18,t] + bus13_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1[2]* model.inertia_constant[18]*model.genD_Pmax[18]*model.ug[18,t]) + bus13_seg1[3] <= model.t13_5[1,t]
model.bus13_5_l1 = Constraint(model.PERIOD, rule = bus13_5_l1)

def bus13_5_u1(model,t):
    return bus13_seg1[0]*100*model.Pg[18,t]*model.ug[18,t] + bus13_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1[2]* model.inertia_constant[18]*model.genD_Pmax[18]*model.ug[18,t]) + bus13_seg1[3] + model.v13_5[1,t]*A >= model.t13_5[1,t]
model.bus13_5_u1 = Constraint(model.PERIOD, rule = bus13_5_u1)

def bus13_5_l2(model,t):
    return bus13_seg2[0]*100*model.Pg[18,t]*model.ug[18,t] + bus13_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2[2]* model.inertia_constant[18]*model.genD_Pmax[18]*model.ug[18,t]) + bus13_seg2[3] <= model.t13_5[1,t]
model.bus13_5_l2 = Constraint(model.PERIOD,rule = bus13_5_l2)

def bus13_5_u2(model,t):
    return bus13_seg2[0]*100*model.Pg[18,t]*model.ug[18,t] + bus13_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2[2]* model.inertia_constant[18]*model.genD_Pmax[18]*model.ug[18,t]) + bus13_seg2[3] + (1 - model.v13_5[1,t])*A >= model.t13_5[1,t]
model.bus13_5_u2 = Constraint(model.PERIOD,rule = bus13_5_u2)

def bus13_5_t12(model,t):
    return model.t13_5[1,t] <= model.t13_5[2,t]
model.bus13_5_t12 = Constraint(model.PERIOD,rule = bus13_5_t12)

def bus13_5_t21(model,t):
    return model.t13_5[2,t]<=model.t13_5[1,t] + model.v13_5[2,t]*A
model.bus13_5_t21 = Constraint(model.PERIOD,rule = bus13_5_t21)

def bus13_5_l3(model,t):
    return bus13_seg3[0]*100*model.Pg[18,t]*model.ug[18,t] + bus13_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3[2]* model.inertia_constant[18]*model.genD_Pmax[18]*model.ug[18,t]) + bus13_seg3[3] <= model.t13_5[2,t]
model.bus13_5_l3 = Constraint(model.PERIOD,rule = bus13_5_l3)

def bus13_5_u3(model,t):
    return bus13_seg3[0]*100*model.Pg[18,t]*model.ug[18,t] + bus13_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3[2]* model.inertia_constant[18]*model.genD_Pmax[18]*model.ug[18,t]) + bus13_seg3[3] + (1 - model.v13_5[2,t])*A >= model.t13_5[2,t]
model.bus13_5_u3 = Constraint(model.PERIOD,rule = bus13_5_u3)

def bus13_5_t23(model,t):
    return model.t13_5[2,t] <= model.t13_5[3,t]
model.bus13_5_t23 = Constraint(model.PERIOD,rule = bus13_5_t23)

def bus13_5_t32(model,t):
    return model.t13_5[3,t]<=model.t13_5[2,t] + model.v13_5[3,t]*A
model.bus13_5_t32 = Constraint(model.PERIOD,rule = bus13_5_t32)

def bus13_5_l4(model,t):
    return bus13_seg4[0]*100*model.Pg[18,t]*model.ug[18,t] + bus13_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4[2]* model.inertia_constant[18]*model.genD_Pmax[18]*model.ug[18,t]) + bus13_seg4[3] <= model.t13_5[3,t]
model.bus13_5_l4 = Constraint(model.PERIOD,rule = bus13_5_l4)

def bus13_5_u4(model,t):
    return bus13_seg4[0]*100*model.Pg[18,t]*model.ug[18,t] + bus13_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4[2]* model.inertia_constant[18]*model.genD_Pmax[18]*model.ug[18,t]) + bus13_seg4[3] + (1 - model.v13_5[3,t])*A >= model.t13_5[3,t]
model.bus13_5_u4 = Constraint(model.PERIOD,rule = bus13_5_u4)

def bus13_6_R(model,t):
    return model.t13_6[3,t] <= RoCoF
model.bus13_6_R = Constraint(model.PERIOD, rule = bus13_6_R)

def bus13_6_l1(model,t):
    return bus13_seg1[0]*100*model.Pg[19,t]*model.ug[19,t] + bus13_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1[2]* model.inertia_constant[19]*model.genD_Pmax[19]*model.ug[19,t]) + bus13_seg1[3] <= model.t13_6[1,t]
model.bus13_6_l1 = Constraint(model.PERIOD, rule = bus13_6_l1)

def bus13_6_u1(model,t):
    return bus13_seg1[0]*100*model.Pg[19,t]*model.ug[19,t] + bus13_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1[2]* model.inertia_constant[19]*model.genD_Pmax[19]*model.ug[19,t]) + bus13_seg1[3] + model.v13_6[1,t]*A >= model.t13_6[1,t]
model.bus13_6_u1 = Constraint(model.PERIOD, rule = bus13_6_u1)

def bus13_6_l2(model,t):
    return bus13_seg2[0]*100*model.Pg[19,t]*model.ug[19,t] + bus13_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2[2]* model.inertia_constant[19]*model.genD_Pmax[19]*model.ug[19,t]) + bus13_seg2[3] <= model.t13_6[1,t]
model.bus13_6_l2 = Constraint(model.PERIOD,rule = bus13_6_l2)

def bus13_6_u2(model,t):
    return bus13_seg2[0]*100*model.Pg[19,t]*model.ug[19,t] + bus13_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2[2]* model.inertia_constant[19]*model.genD_Pmax[19]*model.ug[19,t]) + bus13_seg2[3] + (1 - model.v13_6[1,t])*A >= model.t13_6[1,t]
model.bus13_6_u2 = Constraint(model.PERIOD,rule = bus13_6_u2)

def bus13_6_t12(model,t):
    return model.t13_6[1,t] <= model.t13_6[2,t]
model.bus13_6_t12 = Constraint(model.PERIOD,rule = bus13_6_t12)

def bus13_6_t21(model,t):
    return model.t13_6[2,t]<=model.t13_6[1,t] + model.v13_6[2,t]*A
model.bus13_6_t21 = Constraint(model.PERIOD,rule = bus13_6_t21)

def bus13_6_l3(model,t):
    return bus13_seg3[0]*100*model.Pg[19,t]*model.ug[19,t] + bus13_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3[2]* model.inertia_constant[19]*model.genD_Pmax[19]*model.ug[19,t]) + bus13_seg3[3] <= model.t13_6[2,t]
model.bus13_6_l3 = Constraint(model.PERIOD,rule = bus13_6_l3)

def bus13_6_u3(model,t):
    return bus13_seg3[0]*100*model.Pg[19,t]*model.ug[19,t] + bus13_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3[2]* model.inertia_constant[19]*model.genD_Pmax[19]*model.ug[19,t]) + bus13_seg3[3] + (1 - model.v13_6[2,t])*A >= model.t13_6[2,t]
model.bus13_6_u3 = Constraint(model.PERIOD,rule = bus13_6_u3)

def bus13_6_t23(model,t):
    return model.t13_6[2,t] <= model.t13_6[3,t]
model.bus13_6_t23 = Constraint(model.PERIOD,rule = bus13_6_t23)

def bus13_6_t32(model,t):
    return model.t13_6[3,t]<=model.t13_6[2,t] + model.v13_6[3,t]*A
model.bus13_6_t32 = Constraint(model.PERIOD,rule = bus13_6_t32)

def bus13_6_l4(model,t):
    return bus13_seg4[0]*100*model.Pg[19,t]*model.ug[19,t] + bus13_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4[2]* model.inertia_constant[19]*model.genD_Pmax[19]*model.ug[19,t]) + bus13_seg4[3] <= model.t13_6[3,t]
model.bus13_6_l4 = Constraint(model.PERIOD,rule = bus13_6_l4)

def bus13_6_u4(model,t):
    return bus13_seg4[0]*100*model.Pg[19,t]*model.ug[19,t] + bus13_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4[2]* model.inertia_constant[19]*model.genD_Pmax[19]*model.ug[19,t]) + bus13_seg4[3] + (1 - model.v13_6[3,t])*A >= model.t13_6[3,t]
model.bus13_6_u4 = Constraint(model.PERIOD,rule = bus13_6_u4)


# Node 15 locational RoCoF constraints

def bus15_R(model,g,t):
    if g >= 20:
        if g <= 24:
            return model.t15[3,t] <= RoCoF
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_R = Constraint(model.GEND, model.PERIOD,rule = bus15_R)

def bus15_l1(model,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg1[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg1[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg1[3]<= model.t15[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_l1 = Constraint(model.GEND,model.PERIOD,rule = bus15_l1)

def bus15_u1(model,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg1[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg1[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg1[3] + model.v15[1,t]*A >= model.t15[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_u1 = Constraint(model.GEND, model.PERIOD, rule = bus15_u1)

def bus15_l2(model,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg2[3] <= model.t15[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_l2 = Constraint(model.GEND, model.PERIOD,rule = bus15_l2)

def bus15_u2(model,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg2[3] + ( 1 - model.v15[1, t]) * A >= model.t15[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_u2 = Constraint(model.GEND, model.PERIOD, rule = bus15_u2)

def bus15_t12(model,g,t):
    if g >= 20:
        if g <= 24:
            return model.t15[1,t] <= model.t15[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_t12 = Constraint(model.GEND, model.PERIOD,rule = bus15_t12)

def bus15_t21(model,g,t):
    if g >= 20:
        if g <= 24:
            return model.t15[2,t]<=model.t15[1,t] + model.v15[2,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_t21 = Constraint(model.GEND,model.PERIOD, rule = bus15_t21)

def bus15_l3(model,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg3[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg3[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg3[3] <= model.t15[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_l3 = Constraint(model.GEND,model.PERIOD, rule = bus15_l3)

def bus15_u3(model,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg3[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg3[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg3[3] + (1 - model.v15[2,t])*A >= model.t15[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_u3 = Constraint(model.GEND, model.PERIOD,rule = bus15_u3)

def bus15_t23(model,g,t):
    if g >= 20:
        if g <= 24:
            return model.t15[2,t] <= model.t15[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_t23 = Constraint(model.GEND,model.PERIOD, rule = bus15_t23)

def bus15_t32(model,g,t):
    if g >= 20:
        if g <= 24:
            return model.t15[3,t]<=model.t15[2,t] + model.v15[3,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_t32 = Constraint(model.GEND, model.PERIOD,rule = bus15_t32)

def bus15_l4(model,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg4[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg4[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg4[3] <= model.t15[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_l4 = Constraint(model.GEND,model.PERIOD,rule = bus15_l4)

def bus15_u4(model,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg4[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg4[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg4[3] + (1 - model.v15[3,t])*A >= model.t15[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_u4 = Constraint(model.GEND,model.PERIOD,rule = bus15_u4)

# Node 15_1 constraint
def bus15_1_R(model,g,t):
    if g >= 25:
        if g <= 26:
            return model.t15_1[3,t] <= RoCoF
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_R = Constraint(model.GEND, model.PERIOD,rule = bus15_1_R)

def bus15_1_l1(model,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg1[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg1[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg1[3] <= model.t15_1[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_l1 = Constraint(model.GEND,model.PERIOD,rule = bus15_1_l1)

def bus15_1_u1(model,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg1[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg1[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg1[3]+ model.v15_1[1,t]*A >= model.t15_1[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_u1 = Constraint(model.GEND, model.PERIOD, rule = bus15_1_u1)

def bus15_1_l2(model,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg2[3] <= model.t15_1[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_l2 = Constraint(model.GEND, model.PERIOD,rule = bus15_1_l2)

def bus15_1_u2(model,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg2[3] + ( 1 - model.v15_1[1, t]) * A >= model.t15_1[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_u2 = Constraint(model.GEND, model.PERIOD, rule = bus15_1_u2)

def bus15_1_t12(model,g,t):
    if g >= 25:
        if g <= 26:
            return model.t15_1[1,t] <= model.t15_1[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_t12 = Constraint(model.GEND, model.PERIOD,rule = bus15_1_t12)

def bus15_1_t21(model,g,t):
    if g >= 25:
        if g <= 26:
            return model.t15_1[2,t]<=model.t15_1[1,t] + model.v15_1[2,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_t21 = Constraint(model.GEND,model.PERIOD, rule = bus15_1_t21)

def bus15_1_l3(model,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg3[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg3[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg3[3] <= model.t15_1[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_l3 = Constraint(model.GEND,model.PERIOD, rule = bus15_1_l3)

def bus15_1_u3(model,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg3[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg3[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg3[3] + (1 - model.v15_1[2,t])*A >= model.t15_1[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_u3 = Constraint(model.GEND, model.PERIOD,rule = bus15_1_u3)

def bus15_1_t23(model,g,t):
    if g >= 25:
        if g <= 26:
            return model.t15_1[2,t] <= model.t15_1[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_t23 = Constraint(model.GEND,model.PERIOD, rule = bus15_1_t23)

def bus15_1_t32(model,g,t):
    if g >= 25:
        if g <= 26:
            return model.t15_1[3,t]<=model.t15_1[2,t] + model.v15_1[3,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_t32 = Constraint(model.GEND, model.PERIOD,rule = bus15_1_t32)

def bus15_1_l4(model,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg4[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg4[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg4[3] <= model.t15_1[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_l4 = Constraint(model.GEND,model.PERIOD,rule = bus15_1_l4)

def bus15_1_u4(model,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg4[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg4[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg4[3] + (1 - model.v15_1[3,t])*A >= model.t15_1[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_u4 = Constraint(model.GEND,model.PERIOD,rule = bus15_1_u4)

# Node 16 locational RoCoF constraints

def bus16_R(model,g,t):
    if g >= 27:
        if g <= 29:
            return model.t16[3,t] <= RoCoF
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_R = Constraint(model.GEND, model.PERIOD,rule = bus16_R)

def bus16_l1(model,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg1[0]*100*model.Pg[g,t]*model.ug[g,t] + bus16_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus16_seg1[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus16_seg1[3] <= model.t16[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_l1 = Constraint(model.GEND,model.PERIOD,rule = bus16_l1)

def bus16_u1(model,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg1[0]*100*model.Pg[g,t]*model.ug[g,t] + bus16_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus16_seg1[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus16_seg1[3]+ model.v16[1,t]*A >= model.t16[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_u1 = Constraint(model.GEND, model.PERIOD, rule = bus16_u1)

def bus16_l2(model,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus16_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus16_seg2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus16_seg2[3]<= model.t16[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_l2 = Constraint(model.GEND, model.PERIOD,rule = bus16_l2)

def bus16_u2(model,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus16_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus16_seg2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus16_seg2[3] + ( 1 - model.v16[1, t]) * A >= model.t16[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_u2 = Constraint(model.GEND, model.PERIOD, rule = bus16_u2)

def bus16_t12(model,g,t):
    if g >= 27:
        if g <= 29:
            return model.t16[1,t] <= model.t16[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_t12 = Constraint(model.GEND, model.PERIOD,rule = bus16_t12)

def bus16_t21(model,g,t):
    if g >= 27:
        if g <= 29:
            return model.t16[2,t]<=model.t16[1,t] + model.v16[2,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_t21 = Constraint(model.GEND,model.PERIOD, rule = bus16_t21)

def bus16_l3(model,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg3[0]*100*model.Pg[g,t]*model.ug[g,t] + bus16_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus16_seg3[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus16_seg3[3] <= model.t16[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_l3 = Constraint(model.GEND,model.PERIOD, rule = bus16_l3)

def bus16_u3(model,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg3[0]*100*model.Pg[g,t]*model.ug[g,t] + bus16_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus16_seg3[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus16_seg3[3] + (1 - model.v16[2,t])*A >= model.t16[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_u3 = Constraint(model.GEND, model.PERIOD,rule = bus16_u3)

def bus16_t23(model,g,t):
    if g >= 27:
        if g <= 29:
            return model.t16[2,t] <= model.t16[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_t23 = Constraint(model.GEND,model.PERIOD, rule = bus16_t23)

def bus16_t32(model,g,t):
    if g >= 27:
        if g <= 29:
            return model.t16[3,t]<=model.t16[2,t] + model.v16[3,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_t32 = Constraint(model.GEND, model.PERIOD,rule = bus16_t32)

def bus16_l4(model,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg4[0]*100*model.Pg[g,t]*model.ug[g,t] + bus16_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus16_seg4[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus16_seg4[3] <= model.t16[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_l4 = Constraint(model.GEND,model.PERIOD,rule = bus16_l4)

def bus16_u4(model,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg4[0]*100*model.Pg[g,t]*model.ug[g,t] + bus16_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus16_seg4[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus16_seg4[3] + (1 - model.v16[3,t])*A >= model.t16[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_u4 = Constraint(model.GEND,model.PERIOD,rule = bus16_u4)

# Node 18 locational RoCoF constraints

def bus18_R(model,t):
    return model.t18[3,t] <= RoCoF
model.bus18_R = Constraint(model.PERIOD, rule = bus18_R)

def bus18_l1(model,t):
    return bus18_seg1[0]*100*model.Pg[30,t]*model.ug[30,t] + bus18_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus18_seg1[2]* model.inertia_constant[30]*model.genD_Pmax[30]*model.ug[30,t]) + bus18_seg1[3] <= model.t18[1,t]
model.bus18_l1 = Constraint(model.PERIOD, rule = bus18_l1)

def bus18_u1(model,t):
    return bus18_seg1[0]*100*model.Pg[30,t]*model.ug[30,t] + bus18_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus18_seg1[2]* model.inertia_constant[30]*model.genD_Pmax[30]*model.ug[30,t]) + bus18_seg1[3] + model.v18[1,t]*A >= model.t18[1,t]
model.bus18_u1 = Constraint(model.PERIOD, rule = bus18_u1)

def bus18_l2(model,t):
    return bus18_seg2[0]*100*model.Pg[30,t]*model.ug[30,t] + bus18_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus18_seg2[2]* model.inertia_constant[30]*model.genD_Pmax[30]*model.ug[30,t]) + bus18_seg2[3] <= model.t18[1,t]
model.bus18_l2 = Constraint(model.PERIOD,rule = bus18_l2)

def bus18_u2(model,t):
    return bus18_seg2[0]*100*model.Pg[30,t]*model.ug[30,t] + bus18_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus18_seg2[2]* model.inertia_constant[30]*model.genD_Pmax[30]*model.ug[30,t]) + bus18_seg2[3]+ (1 - model.v18[1,t])*A >= model.t18[1,t]
model.bus18_u2 = Constraint(model.PERIOD,rule = bus18_u2)

def bus18_t12(model,t):
    return model.t18[1,t] <= model.t18[2,t]
model.bus18_t12 = Constraint(model.PERIOD,rule = bus18_t12)

def bus18_t21(model,t):
    return model.t18[2,t]<=model.t18[1,t] + model.v18[2,t]*A
model.bus18_t21 = Constraint(model.PERIOD,rule = bus18_t21)

def bus18_l3(model,t):
    return bus18_seg3[0]*100*model.Pg[30,t]*model.ug[30,t] + bus18_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus18_seg3[2]* model.inertia_constant[30]*model.genD_Pmax[30]*model.ug[30,t]) + bus18_seg3[3]<= model.t18[2,t]
model.bus18_l3 = Constraint(model.PERIOD,rule = bus18_l3)

def bus18_u3(model,t):
    return bus18_seg3[0]*100*model.Pg[30,t]*model.ug[30,t] + bus18_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus18_seg3[2]* model.inertia_constant[30]*model.genD_Pmax[30]*model.ug[30,t]) + bus18_seg3[3] + (1 - model.v18[2,t])*A >= model.t18[2,t]
model.bus18_u3 = Constraint(model.PERIOD,rule = bus18_u3)

def bus18_t23(model,t):
    return model.t18[2,t] <= model.t18[3,t]
model.bus18_t23 = Constraint(model.PERIOD,rule = bus18_t23)

def bus18_t32(model,t):
    return model.t18[3,t]<=model.t18[2,t] + model.v18[3,t]*A
model.bus18_t32 = Constraint(model.PERIOD,rule = bus18_t32)

def bus18_l4(model,t):
    return bus18_seg4[0]*100*model.Pg[30,t]*model.ug[30,t] + bus18_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus18_seg4[2]* model.inertia_constant[30]*model.genD_Pmax[30]*model.ug[30,t]) + bus18_seg4[3] <= model.t18[3,t]
model.bus18_l4 = Constraint(model.PERIOD,rule = bus18_l4)

def bus18_u4(model,t):
    return bus18_seg4[0]*100*model.Pg[30,t]*model.ug[30,t] + bus18_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus18_seg4[2]* model.inertia_constant[30]*model.genD_Pmax[30]*model.ug[30,t]) + bus18_seg4[3] + (1 - model.v18[3,t])*A >= model.t18[3,t]
model.bus18_u4 = Constraint(model.PERIOD,rule = bus18_u4)

# Node 21 locational RoCoF constraints

def bus21_R(model,g,t):
    if g >= 31:
        if g <= 35:
            return model.t21[3,t] <= RoCoF
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_R = Constraint(model.GEND, model.PERIOD,rule = bus21_R)

def bus21_l1(model,g,t):
    if g >= 31:
        if g <= 35:
            return 0.0027*100*model.Pg[g,t]*model.ug[g,t]- 0.000001*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) - model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) <= model.t21[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_l1 = Constraint(model.GEND,model.PERIOD,rule = bus21_l1)

def bus21_u1(model,g,t):
    if g >= 31:
        if g <= 35:
            return 0.0027*100*model.Pg[g,t]*model.ug[g,t]- 0.000001*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) - model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) + model.v21[1,t]*A >= model.t21[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_u1 = Constraint(model.GEND, model.PERIOD, rule = bus21_u1)

def bus21_l2(model,g,t):
    if g >= 31:
        if g <= 35:
            return 0.0013*100*model.Pg[g,t]*model.ug[g,t]- 0.000001*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) - model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t])  + 0.3457 <= model.t21[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_l2 = Constraint(model.GEND, model.PERIOD,rule = bus21_l2)

def bus21_u2(model,g,t):
    if g >= 31:
        if g <= 35:
            return 0.0013 * 100 * model.Pg[g, t] * model.ug[g, t] - 0.000001*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) - model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t])+ 0.3457 + ( 1 - model.v21[1, t]) * A >= model.t21[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_u2 = Constraint(model.GEND, model.PERIOD, rule = bus21_u2)

def bus21_t12(model,g,t):
    if g >= 31:
        if g <= 35:
            return model.t21[1,t] <= model.t21[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_t12 = Constraint(model.GEND, model.PERIOD,rule = bus21_t12)

def bus21_t21(model,g,t):
    if g >= 31:
        if g <= 35:
            return model.t21[2,t]<=model.t21[1,t] + model.v21[2,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_t21 = Constraint(model.GEND,model.PERIOD, rule = bus21_t21)

def bus21_l3(model,g,t):
    if g >= 31:
        if g <= 35:
            return 0.0027*100*model.Pg[g,t]*model.ug[g,t] - 0.000001*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) - model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) <= model.t21[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_l3 = Constraint(model.GEND,model.PERIOD, rule = bus21_l3)

def bus21_u3(model,g,t):
    if g >= 31:
        if g <= 35:
            return 0.0027*100*model.Pg[g,t]*model.ug[g,t] - 0.000001*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) - model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) + (1 - model.v21[2,t])*A >= model.t21[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_u3 = Constraint(model.GEND, model.PERIOD,rule = bus21_u3)

def bus21_t23(model,g,t):
    if g >= 31:
        if g <= 35:
            return model.t21[2,t] <= model.t21[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_t23 = Constraint(model.GEND,model.PERIOD, rule = bus21_t23)

def bus21_t32(model,g,t):
    if g >= 31:
        if g <= 35:
            return model.t21[3,t]<=model.t21[2,t] + model.v21[3,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_t32 = Constraint(model.GEND, model.PERIOD,rule = bus21_t32)

def bus21_l4(model,g,t):
    if g >= 31:
        if g <= 35:
            return 0 <= model.t21[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_l4 = Constraint(model.GEND,model.PERIOD,rule = bus21_l4)

def bus21_u4(model,g,t):
    if g >= 31:
        if g <= 35:
            return (1 - model.v21[3,t])*A >= model.t21[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_u4 = Constraint(model.GEND,model.PERIOD,rule = bus21_u4)


# Node 22 constraints
def bus22_R(model,t):
    return model.t22[3,t] <= RoCoF
model.bus22_R = Constraint(model.PERIOD, rule = bus22_R)

def bus22_l1(model,t):
    return bus22_seg1[0]*100*model.Pg[36,t]*model.ug[36,t] + bus22_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus22_seg1[2]* model.inertia_constant[36]*model.genD_Pmax[36]*model.ug[36,t]) + bus22_seg1[3] <= model.t22[1,t]
model.bus22_l1 = Constraint(model.PERIOD, rule = bus22_l1)

def bus22_u1(model,t):
    return bus22_seg1[0]*100*model.Pg[36,t]*model.ug[36,t] + bus22_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus22_seg1[2]* model.inertia_constant[36]*model.genD_Pmax[36]*model.ug[36,t]) + bus22_seg1[3] + model.v22[1,t]*A >= model.t22[1,t]
model.bus22_u1 = Constraint(model.PERIOD, rule = bus22_u1)

def bus22_l2(model,t):
    return bus22_seg2[0]*100*model.Pg[36,t]*model.ug[36,t] + bus22_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus22_seg2[2]* model.inertia_constant[36]*model.genD_Pmax[36]*model.ug[36,t]) + bus22_seg2[3] <= model.t22[1,t]
model.bus22_l2 = Constraint(model.PERIOD,rule = bus22_l2)

def bus22_u2(model,t):
    return bus22_seg2[0]*100*model.Pg[36,t]*model.ug[36,t] + bus22_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus22_seg2[2]* model.inertia_constant[36]*model.genD_Pmax[36]*model.ug[36,t]) + bus22_seg2[3] + (1 - model.v22[1,t])*A >= model.t22[1,t]
model.bus22_u2 = Constraint(model.PERIOD,rule = bus22_u2)

def bus22_t12(model,t):
    return model.t22[1,t] <= model.t22[2,t]
model.bus22_t12 = Constraint(model.PERIOD,rule = bus22_t12)

def bus22_t21(model,t):
    return model.t22[2,t]<=model.t22[1,t] + model.v22[2,t]*A
model.bus22_t21 = Constraint(model.PERIOD,rule = bus22_t21)

def bus22_l3(model,t):
    return bus22_seg3[0]*100*model.Pg[36,t]*model.ug[36,t] + bus22_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus22_seg3[2]* model.inertia_constant[36]*model.genD_Pmax[36]*model.ug[36,t]) + bus22_seg3[3] <= model.t22[2,t]
model.bus22_l3 = Constraint(model.PERIOD,rule = bus22_l3)

def bus22_u3(model,t):
    return bus22_seg3[0]*100*model.Pg[36,t]*model.ug[36,t] + bus22_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus22_seg3[2]* model.inertia_constant[36]*model.genD_Pmax[36]*model.ug[36,t]) + bus22_seg3[3] + (1 - model.v22[2,t])*A >= model.t22[2,t]
model.bus22_u3 = Constraint(model.PERIOD,rule = bus22_u3)

def bus22_t23(model,t):
    return model.t22[2,t] <= model.t22[3,t]
model.bus22_t23 = Constraint(model.PERIOD,rule = bus22_t23)

def bus22_t32(model,t):
    return model.t22[3,t]<=model.t22[2,t] + model.v22[3,t]*A
model.bus22_t32 = Constraint(model.PERIOD,rule = bus22_t32)

def bus22_l4(model,t):
    return bus22_seg4[0]*100*model.Pg[36,t]*model.ug[36,t] + bus22_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus22_seg4[2]* model.inertia_constant[36]*model.genD_Pmax[36]*model.ug[36,t]) + bus22_seg4[3] <= model.t22[3,t]
model.bus22_l4 = Constraint(model.PERIOD,rule = bus22_l4)

def bus22_u4(model,t):
    return bus22_seg4[0]*100*model.Pg[36,t]*model.ug[36,t] + bus22_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus22_seg4[2]* model.inertia_constant[36]*model.genD_Pmax[36]*model.ug[36,t]) + bus22_seg4[3] + (1 - model.v22[3,t])*A >= model.t22[3,t]
model.bus22_u4 = Constraint(model.PERIOD,rule = bus22_u4)


# Node 23 locational RoCoF constraints

def bus23_R(model,g,t):
    if g >= 39:
        if g <= 41:
            return model.t23[3,t] <= RoCoF
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_R = Constraint(model.GEND, model.PERIOD,rule = bus23_R)

def bus23_l1(model,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg1[0]*100*model.Pg[g,t]*model.ug[g,t] + bus23_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus23_seg1[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus23_seg1[3]  <= model.t23[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_l1 = Constraint(model.GEND,model.PERIOD,rule = bus23_l1)

def bus23_u1(model,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg1[0]*100*model.Pg[g,t]*model.ug[g,t] + bus23_seg1[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus23_seg1[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus23_seg1[3]  + model.v23[1,t]*A >= model.t23[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_u1 = Constraint(model.GEND, model.PERIOD, rule = bus23_u1)

def bus23_l2(model,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus23_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus23_seg2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus23_seg2[3]  <= model.t23[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_l2 = Constraint(model.GEND, model.PERIOD,rule = bus23_l2)

def bus23_u2(model,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus23_seg2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus23_seg2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus23_seg2[3]  + ( 1 - model.v23[1, t]) * A >= model.t23[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_u2 = Constraint(model.GEND, model.PERIOD, rule = bus23_u2)

def bus23_t12(model,g,t):
    if g >= 39:
        if g <= 41:
            return model.t23[1,t] <= model.t23[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_t12 = Constraint(model.GEND, model.PERIOD,rule = bus23_t12)

def bus23_t21(model,g,t):
    if g >= 39:
        if g <= 41:
            return model.t23[2,t]<=model.t23[1,t] + model.v23[2,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_t21 = Constraint(model.GEND,model.PERIOD, rule = bus23_t21)

def bus23_l3(model,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg3[0]*100*model.Pg[g,t]*model.ug[g,t] + bus23_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus23_seg3[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus23_seg3[3]  <= model.t23[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_l3 = Constraint(model.GEND,model.PERIOD, rule = bus23_l3)

def bus23_u3(model,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg3[0]*100*model.Pg[g,t]*model.ug[g,t] + bus23_seg3[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus23_seg3[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus23_seg3[3]  + (1 - model.v23[2,t])*A >= model.t23[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_u3 = Constraint(model.GEND, model.PERIOD,rule = bus23_u3)

def bus23_t23(model,g,t):
    if g >= 39:
        if g <= 41:
            return model.t23[2,t] <= model.t23[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_t23 = Constraint(model.GEND,model.PERIOD, rule = bus23_t23)

def bus23_t32(model,g,t):
    if g >= 39:
        if g <= 41:
            return model.t23[3,t]<=model.t23[2,t] + model.v23[3,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_t32 = Constraint(model.GEND, model.PERIOD,rule = bus23_t32)

def bus23_l4(model,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg4[0]*100*model.Pg[g,t]*model.ug[g,t] + bus23_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus23_seg4[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus23_seg4[3]  <= model.t23[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_l4 = Constraint(model.GEND,model.PERIOD,rule = bus23_l4)

def bus23_u4(model,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg4[0]*100*model.Pg[g,t]*model.ug[g,t] + bus23_seg4[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus23_seg4[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus23_seg4[3]  + (1 - model.v23[3,t])*A >= model.t23[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_u4 = Constraint(model.GEND,model.PERIOD,rule = bus23_u4)


def bus01_R_t2(model,t):
    return model.t01_t2[3,t] <= RoCoF
model.bus01_R_t2 = Constraint(model.PERIOD,rule = bus01_R_t2)

def bus01_l1_t2(model,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg1_t2[0]*100*model.Pg[g,t]*model.ug[g,t]+ bus1_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND)) + bus1_seg1_t2[2] * model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t] + bus1_seg1_t2[3] <= model.t01_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus01_l1_t2 = Constraint(model.GEND,model.PERIOD,rule = bus01_l1_t2)

def bus01_u1_t2(model,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg1_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus1_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus1_seg1_t2[2] * model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) + bus1_seg1_t2[3] + model.v01_t2[1,t]*A >= model.t01_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus01_u1_t2 = Constraint(model.GEND, model.PERIOD, rule = bus01_u1_t2)

def bus01_l2_t2(model,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg2_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus1_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND)) + bus1_seg2_t2[2] * model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t] +  bus1_seg2_t2[3] <= model.t01_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus01_l2_t2 = Constraint(model.GEND, model.PERIOD,rule = bus01_l2_t2)

def bus01_u2_t2(model,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg2_t2[0]*100*model.Pg[g,t]*model.ug[g,t] +  bus1_seg2_t2[1]/(fn*np.pi)*(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND)) + bus1_seg2_t2[2] * model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]+  bus1_seg2_t2[3] + ( 1 - model.v01_t2[1, t]) * A >= model.t01_t2[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus01_u2_t2 = Constraint(model.GEND, model.PERIOD, rule = bus01_u2_t2)

def bus01_t12_t2(model,t):
    return model.t01_t2[1,t] <= model.t01_t2[2,t]
model.bus01_t12_t2 = Constraint( model.PERIOD,rule = bus01_t12_t2)

def bus01_t21_t2(model,t):
    return model.t01_t2[2,t]<=model.t01_t2[1,t] + model.v01_t2[2,t]*A
model.bus01_t21_t2 = Constraint(model.PERIOD, rule = bus01_t21_t2)

def bus01_l3_t2(model,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg3_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus1_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus1_seg3_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus1_seg3_t2[3]<= model.t01_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus01_l3_t2 = Constraint(model.GEND,model.PERIOD, rule = bus01_l3_t2)

def bus01_u3_t2(model,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg3_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus1_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND)) + bus1_seg3_t2[2]* model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]  + bus1_seg3_t2[3] + (1 - model.v01_t2[2,t])*A >= model.t01_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus01_u3_t2 = Constraint(model.GEND, model.PERIOD,rule = bus01_u3_t2)

def bus01_t23_t2(model,t):
    return model.t01_t2[2,t] <= model.t01_t2[3,t]
model.bus01_t23_t2 = Constraint(model.PERIOD, rule = bus01_t23_t2)

def bus01_t32_t2(model,t):
    return model.t01_t2[3,t]<=model.t01_t2[2,t] + model.v01_t2[3,t]*A
model.bus01_t32_t2 = Constraint(model.PERIOD,rule = bus01_t32_t2)

def bus01_l4_t2(model,g,t):
    if g >= 1:
        if g <= 4:
             return bus1_seg4_t2[0]*model.Pg[g,t]*model.ug[g,t] + bus1_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND)) + bus1_seg4_t2[2]* model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t] + bus1_seg4_t2[3] <= model.t01_t2[3,t]
        else:
             return Constraint.Skip
    else:
        return Constraint.Skip
model.bus01_l4_t2 = Constraint(model.GEND,model.PERIOD,rule = bus01_l4_t2)

def bus01_u4_t2(model,g,t):
    if g >= 1:
        if g <= 4:
             return bus1_seg4_t2[0]*model.Pg[g,t]*model.ug[g,t] + bus1_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND)) + bus1_seg4_t2[2]* model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t] + bus1_seg4_t2[3] +(1 - model.v01_t2[3,t])*A >= model.t01_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus01_u4_t2 = Constraint(model.GEND,model.PERIOD,rule = bus01_u4_t2)

# Node 02 locational RoCoF constraints

def bus02_1_R_t2(model,t):
    return model.t02_1_t2[3,t] <= RoCoF
model.bus02_1_R_t2 = Constraint(model.PERIOD, rule = bus02_1_R_t2)

def bus02_1_l1_t2(model,t):
    return bus2_seg1_t2[0]*100*model.Pg[5,t]*model.ug[5,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1_t2[2] * model.inertia_constant[5]*model.genD_Pmax[5]*model.ug[5,t]) + bus2_seg1_t2[3] <= model.t02_1_t2[1,t]
model.bus02_1_l1_t2 = Constraint(model.PERIOD, rule = bus02_1_l1_t2)

def bus02_1_u1_t2(model,t):
    return bus2_seg1_t2[0]*100*model.Pg[5,t]*model.ug[5,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1_t2[2] * model.inertia_constant[5]*model.genD_Pmax[5]*model.ug[5,t]) + bus2_seg1_t2[3] + model.v02_1_t2[1,t]*A >= model.t02_1_t2[1,t]
model.bus02_1_u1_t2 = Constraint(model.PERIOD, rule = bus02_1_u1_t2)

def bus02_1_l2_t2(model,t):
    return bus2_seg2_t2[0]*100*model.Pg[5,t]*model.ug[5,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2_t2[2] * model.inertia_constant[5]*model.genD_Pmax[5]*model.ug[5,t]) + bus2_seg2_t2[3] <= model.t02_1_t2[1,t]
model.bus02_1_l2_t2 = Constraint(model.PERIOD,rule = bus02_1_l2_t2)

def bus02_1_u2_t2(model,t):
    return bus2_seg2_t2[0]*100*model.Pg[5,t]*model.ug[5,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2_t2[2] * model.inertia_constant[5]*model.genD_Pmax[5]*model.ug[5,t]) + bus2_seg2_t2[3] + (1 - model.v02_1_t2[1,t])*A >= model.t02_1_t2[1,t]
model.bus02_1_u2_t2 = Constraint(model.PERIOD,rule = bus02_1_u2_t2)

def bus02_1_t12_t2(model,t):
    return model.t02_1_t2[1,t] <= model.t02_1_t2[2,t]
model.bus02_1_t12_t2 = Constraint(model.PERIOD,rule = bus02_1_t12_t2)

def bus02_1_t21_t2(model,t):
    return model.t02_1_t2[2,t]<=model.t02_1_t2[1,t] + model.v02_1_t2[2,t]*A
model.bus02_1_t21_t2 = Constraint(model.PERIOD,rule = bus02_1_t21_t2)

def bus02_1_l3_t2(model,t):
    return bus2_seg3_t2[0]*100*model.Pg[5,t]*model.ug[5,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3_t2[2]* model.inertia_constant[5]*model.genD_Pmax[5]*model.ug[5,t]) + bus2_seg3_t2[3]  <= model.t02_1_t2[2,t]
model.bus02_1_l3_t2 = Constraint(model.PERIOD,rule = bus02_1_l3_t2)

def bus02_1_u3_t2(model,t):
    return bus2_seg3_t2[0]*100*model.Pg[5,t]*model.ug[5,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3_t2[2]* model.inertia_constant[5]*model.genD_Pmax[5]*model.ug[5,t]) + bus2_seg3_t2[3]   + (1 - model.v02_1_t2[2,t])*A >= model.t02_1_t2[2,t]
model.bus02_1_u3_t2 = Constraint(model.PERIOD,rule = bus02_1_u3_t2)

def bus02_1_t23_t2(model,t):
    return model.t02_1_t2[2,t] <= model.t02_1_t2[3,t]
model.bus02_1_t23_t2 = Constraint(model.PERIOD,rule = bus02_1_t23_t2)

def bus02_1_t32_t2(model,t):
    return model.t02_1_t2[3,t]<=model.t02_1_t2[2,t] + model.v02_1_t2[3,t]*A
model.bus02_1_t32_t2 = Constraint(model.PERIOD,rule = bus02_1_t32_t2)

def bus02_1_l4_t2(model,t):
    return bus2_seg4_t2[0]*100*model.Pg[5,t]*model.ug[5,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4_t2[2]* model.inertia_constant[5]*model.genD_Pmax[5]*model.ug[5,t]) + bus2_seg4_t2[3]  <= model.t02_1_t2[3,t]
model.bus02_1_l4_t2 = Constraint(model.PERIOD,rule = bus02_1_l4_t2)

def bus02_1_u4_t2(model,t):
    return bus2_seg4_t2[0]*100*model.Pg[5,t]*model.ug[5,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4_t2[2]* model.inertia_constant[5]*model.genD_Pmax[5]*model.ug[5,t]) + bus2_seg4_t2[3]+(1 - model.v02_1_t2[3,t])*A >= model.t02_1_t2[3,t]
model.bus02_1_u4_t2 = Constraint(model.PERIOD,rule = bus02_1_u4_t2)

def bus02_2_R_t2(model,t):
    return model.t02_2_t2[3,t] <= RoCoF
model.bus02_2_R_t2 = Constraint(model.PERIOD, rule = bus02_2_R_t2)

def bus02_2_l1_t2(model,t):
    return bus2_seg1_t2[0]*100*model.Pg[6,t]*model.ug[6,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1_t2[2]* model.inertia_constant[6]*model.genD_Pmax[6]*model.ug[6,t]) + bus2_seg1_t2[3]<= model.t02_2_t2[1,t]
model.bus02_2_l1_t2 = Constraint(model.PERIOD, rule = bus02_2_l1_t2)

def bus02_2_u1_t2(model,t):
    return bus2_seg1_t2[0]*100*model.Pg[6,t]*model.ug[6,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1_t2[2]* model.inertia_constant[6]*model.genD_Pmax[6]*model.ug[6,t]) + bus2_seg1_t2[3] + model.v02_2_t2[1,t]*A >= model.t02_2_t2[1,t]
model.bus02_2_u1_t2 = Constraint(model.PERIOD, rule = bus02_2_u1_t2)

def bus02_2_l2_t2(model,t):
    return bus2_seg2_t2[0]*100*model.Pg[6,t]*model.ug[6,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2_t2[2]* model.inertia_constant[6]*model.genD_Pmax[6]*model.ug[6,t]) + bus2_seg2_t2[3] <= model.t02_2_t2[1,t]
model.bus02_2_l2_t2 = Constraint(model.PERIOD,rule = bus02_2_l2_t2)

def bus02_2_u2_t2(model,t):
    return bus2_seg2_t2[0]*100*model.Pg[6,t]*model.ug[6,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2_t2[2]* model.inertia_constant[6]*model.genD_Pmax[6]*model.ug[6,t]) + bus2_seg2_t2[3]+ (1 - model.v02_2_t2[1,t])*A >= model.t02_2_t2[1,t]
model.bus02_2_u2_t2 = Constraint(model.PERIOD,rule = bus02_2_u2_t2)

def bus02_2_t12_t2(model,t):
    return model.t02_2_t2[1,t] <= model.t02_2_t2[2,t]
model.bus02_2_t12_t2 = Constraint(model.PERIOD,rule = bus02_2_t12_t2)

def bus02_2_t21_t2(model,t):
    return model.t02_2_t2[2,t]<=model.t02_2_t2[1,t] + model.v02_2_t2[2,t]*A
model.bus02_2_t21_t2 = Constraint(model.PERIOD,rule = bus02_2_t21_t2)

def bus02_2_l3_t2(model,t):
    return bus2_seg3_t2[0]*100*model.Pg[6,t]*model.ug[6,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3_t2[2]* model.inertia_constant[6]*model.genD_Pmax[6]*model.ug[6,t]) + bus2_seg3_t2[3] <= model.t02_2_t2[2,t]
model.bus02_2_l3_t2 = Constraint(model.PERIOD,rule = bus02_2_l3_t2)

def bus02_2_u3_t2(model,t):
    return bus2_seg3_t2[0]*100*model.Pg[6,t]*model.ug[6,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3_t2[2]* model.inertia_constant[6]*model.genD_Pmax[6]*model.ug[6,t]) + bus2_seg3_t2[3] + (1 - model.v02_2_t2[2,t])*A >= model.t02_2_t2[2,t]
model.bus02_2_u3_t2 = Constraint(model.PERIOD,rule = bus02_2_u3_t2)

def bus02_2_t23_t2(model,t):
    return model.t02_2_t2[2,t] <= model.t02_2_t2[3,t]
model.bus02_2_t23_t2 = Constraint(model.PERIOD,rule = bus02_2_t23_t2)

def bus02_2_t32_t2(model,t):
    return model.t02_2_t2[3,t]<=model.t02_2_t2[2,t] + model.v02_2_t2[3,t]*A
model.bus02_2_t32_t2 = Constraint(model.PERIOD,rule = bus02_2_t32_t2)

def bus02_2_l4_t2(model,t):
    return bus2_seg4_t2[0]*100*model.Pg[6,t]*model.ug[6,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4_t2[2]* model.inertia_constant[6]*model.genD_Pmax[6]*model.ug[6,t]) + bus2_seg4_t2[3] <= model.t02_2_t2[3,t]
model.bus02_2_l4_t2 = Constraint(model.PERIOD,rule = bus02_2_l4_t2)

def bus02_2_u4_t2(model,t):
    return bus2_seg4_t2[0]*100*model.Pg[6,t]*model.ug[6,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4_t2[2]* model.inertia_constant[6]*model.genD_Pmax[6]*model.ug[6,t]) + bus2_seg4_t2[3] + (1 - model.v02_2_t2[3,t])*A >= model.t02_2_t2[3,t]
model.bus02_2_u4_t2 = Constraint(model.PERIOD,rule = bus02_2_u4_t2)

def bus02_3_R_t2(model,t):
    return model.t02_3_t2[3,t] <= RoCoF
model.bus02_3_R_t2 = Constraint(model.PERIOD, rule = bus02_3_R_t2)

def bus02_3_l1_t2(model,t):
    return bus2_seg1_t2[0]*100*model.Pg[7,t]*model.ug[7,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1_t2[2]* model.inertia_constant[7]*model.genD_Pmax[7]*model.ug[7,t]) + bus2_seg1_t2[3] <= model.t02_3_t2[1,t]
model.bus02_3_l1_t2 = Constraint(model.PERIOD, rule = bus02_3_l1_t2)

def bus02_3_u1_t2(model,t):
    return bus2_seg1_t2[0]*100*model.Pg[7,t]*model.ug[7,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1_t2[2]* model.inertia_constant[7]*model.genD_Pmax[7]*model.ug[7,t]) + bus2_seg1_t2[3] + model.v02_3_t2[1,t]*A >= model.t02_3_t2[1,t]
model.bus02_3_u1_t2 = Constraint(model.PERIOD, rule = bus02_3_u1_t2)

def bus02_3_l2_t2(model,t):
    return bus2_seg2_t2[0]*100*model.Pg[7,t]*model.ug[7,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2_t2[2]* model.inertia_constant[7]*model.genD_Pmax[7]*model.ug[7,t]) + bus2_seg2_t2[3] <= model.t02_3_t2[1,t]
model.bus02_3_l2_t2 = Constraint(model.PERIOD,rule = bus02_3_l2_t2)

def bus02_3_u2_t2(model,t):
    return bus2_seg2_t2[0]*100*model.Pg[7,t]*model.ug[7,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2_t2[2]* model.inertia_constant[7]*model.genD_Pmax[7]*model.ug[7,t]) + bus2_seg2_t2[3]+ (1 - model.v02_3_t2[1,t])*A >= model.t02_3_t2[1,t]
model.bus02_3_u2_t2 = Constraint(model.PERIOD,rule = bus02_3_u2_t2)

def bus02_3_t12_t2(model,t):
    return model.t02_3_t2[1,t] <= model.t02_3_t2[2,t]
model.bus02_3_t12_t2 = Constraint(model.PERIOD,rule = bus02_3_t12_t2)

def bus02_3_t21_t2(model,t):
    return model.t02_3_t2[2,t]<=model.t02_3_t2[1,t] + model.v02_3_t2[2,t]*A
model.bus02_3_t21_t2 = Constraint(model.PERIOD,rule = bus02_3_t21_t2)

def bus02_3_l3_t2(model,t):
    return bus2_seg3_t2[0]*100*model.Pg[7,t]*model.ug[7,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3_t2[2]* model.inertia_constant[7]*model.genD_Pmax[7]*model.ug[7,t]) + bus2_seg3_t2[3]  <= model.t02_3_t2[2,t]
model.bus02_3_l3_t2 = Constraint(model.PERIOD,rule = bus02_3_l3_t2)

def bus02_3_u3_t2(model,t):
    return bus2_seg3_t2[0]*100*model.Pg[7,t]*model.ug[7,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3_t2[2]* model.inertia_constant[7]*model.genD_Pmax[7]*model.ug[7,t]) + bus2_seg3_t2[3] + (1 - model.v02_3_t2[2,t])*A >= model.t02_3_t2[2,t]
model.bus02_3_u3_t2 = Constraint(model.PERIOD,rule = bus02_3_u3_t2)

def bus02_3_t23_t2(model,t):
    return model.t02_3_t2[2,t] <= model.t02_3_t2[3,t]
model.bus02_3_t23_t2 = Constraint(model.PERIOD,rule = bus02_3_t23_t2)

def bus02_3_t32_t2(model,t):
    return model.t02_3_t2[3,t]<=model.t02_3_t2[2,t] + model.v02_3_t2[3,t]*A
model.bus02_3_t32_t2 = Constraint(model.PERIOD,rule = bus02_3_t32_t2)

def bus02_3_l4_t2(model,t):
    return bus2_seg4_t2[0]*100*model.Pg[7,t]*model.ug[7,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4_t2[2]* model.inertia_constant[7]*model.genD_Pmax[7]*model.ug[7,t]) + bus2_seg4_t2[3] <= model.t02_3_t2[3,t]
model.bus02_3_l4_t2 = Constraint(model.PERIOD,rule = bus02_3_l4_t2)

def bus02_3_u4_t2(model,t):
    return bus2_seg4_t2[0]*100*model.Pg[7,t]*model.ug[7,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4_t2[2]* model.inertia_constant[7]*model.genD_Pmax[7]*model.ug[7,t]) + bus2_seg4_t2[3] + (1 - model.v02_3_t2[3,t])*A >= model.t02_3_t2[3,t]
model.bus02_3_u4_t2 = Constraint(model.PERIOD,rule = bus02_3_u4_t2)

def bus02_4_R_t2(model,t):
    return model.t02_4_t2[3,t] <= RoCoF
model.bus02_4_R_t2 = Constraint(model.PERIOD, rule = bus02_4_R_t2)

def bus02_4_l1_t2(model,t):
    return bus2_seg1_t2[0]*100*model.Pg[8,t]*model.ug[8,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1_t2[2]* model.inertia_constant[8]*model.genD_Pmax[8]*model.ug[8,t]) + bus2_seg1_t2[3] <= model.t02_4_t2[1,t]
model.bus02_4_l1_t2 = Constraint(model.PERIOD, rule = bus02_4_l1_t2)

def bus02_4_u1_t2(model,t):
    return bus2_seg1_t2[0]*100*model.Pg[8,t]*model.ug[8,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1_t2[2]* model.inertia_constant[8]*model.genD_Pmax[8]*model.ug[8,t]) + bus2_seg1_t2[3] + model.v02_4_t2[1,t]*A >= model.t02_4_t2[1,t]
model.bus02_4_u1_t2 = Constraint(model.PERIOD, rule = bus02_4_u1_t2)

def bus02_4_l2_t2(model,t):
    return bus2_seg2_t2[0]*100*model.Pg[8,t]*model.ug[8,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2_t2[2]* model.inertia_constant[8]*model.genD_Pmax[8]*model.ug[8,t]) + bus2_seg2_t2[3] <= model.t02_4_t2[1,t]
model.bus02_4_l2_t2 = Constraint(model.PERIOD,rule = bus02_4_l2_t2)

def bus02_4_u2_t2(model,t):
    return bus2_seg2_t2[0]*100*model.Pg[8,t]*model.ug[8,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2_t2[2]* model.inertia_constant[8]*model.genD_Pmax[8]*model.ug[8,t]) + bus2_seg2_t2[3] + (1 - model.v02_4_t2[1,t])*A >= model.t02_4_t2[1,t]
model.bus02_4_u2_t2 = Constraint(model.PERIOD,rule = bus02_4_u2_t2)

def bus02_4_t12_t2(model,t):
    return model.t02_4_t2[1,t] <= model.t02_4_t2[2,t]
model.bus02_4_t12_t2 = Constraint(model.PERIOD,rule = bus02_4_t12_t2)

def bus02_4_t21_t2(model,t):
    return model.t02_4_t2[2,t]<=model.t02_4_t2[1,t] + model.v02_4_t2[2,t]*A
model.bus02_4_t21_t2 = Constraint(model.PERIOD,rule = bus02_4_t21_t2)

def bus02_4_l3_t2(model,t):
    return bus2_seg3_t2[0]*100*model.Pg[8,t]*model.ug[8,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3_t2[2]* model.inertia_constant[8]*model.genD_Pmax[8]*model.ug[8,t]) + bus2_seg3_t2[3]  <= model.t02_4_t2[2,t]
model.bus02_4_l3_t2 = Constraint(model.PERIOD,rule = bus02_4_l3_t2)

def bus02_4_u3_t2(model,t):
    return bus2_seg3_t2[0]*100*model.Pg[8,t]*model.ug[8,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3_t2[2]* model.inertia_constant[8]*model.genD_Pmax[8]*model.ug[8,t]) + bus2_seg3_t2[3] + (1 - model.v02_4_t2[2,t])*A >= model.t02_4_t2[2,t]
model.bus02_4_u3_t2 = Constraint(model.PERIOD,rule = bus02_4_u3_t2)

def bus02_4_t23_t2(model,t):
    return model.t02_4_t2[2,t] <= model.t02_4_t2[3,t]
model.bus02_4_t23_t2 = Constraint(model.PERIOD,rule = bus02_4_t23_t2)

def bus02_4_t32_t2(model,t):
    return model.t02_4_t2[3,t]<=model.t02_4_t2[2,t] + model.v02_4_t2[3,t]*A
model.bus02_4_t32_t2 = Constraint(model.PERIOD,rule = bus02_4_t32_t2)

def bus02_4_l4_t2(model,t):
    return bus2_seg4_t2[0]*100*model.Pg[8,t]*model.ug[8,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4_t2[2]* model.inertia_constant[8]*model.genD_Pmax[8]*model.ug[8,t]) + bus2_seg4_t2[3] <= model.t02_4_t2[3,t]
model.bus02_4_l4_t2 = Constraint(model.PERIOD,rule = bus02_4_l4_t2)

def bus02_4_u4_t2(model,t):
    return bus2_seg4_t2[0]*100*model.Pg[8,t]*model.ug[8,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4_t2[2]* model.inertia_constant[8]*model.genD_Pmax[8]*model.ug[8,t]) + bus2_seg4_t2[3] + (1 - model.v02_4_t2[3,t])*A >= model.t02_4_t2[3,t]
model.bus02_4_u4_t2 = Constraint(model.PERIOD,rule = bus02_4_u4_t2)

def bus02_5_R_t2(model,t):
    return model.t02_5_t2[3,t] <= RoCoF
model.bus02_5_R_t2 = Constraint(model.PERIOD, rule = bus02_5_R_t2)

def bus02_5_l1_t2(model,t):
    return bus2_seg1_t2[0]*100*model.Pg[9,t]*model.ug[9,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1_t2[2]* model.inertia_constant[9]*model.genD_Pmax[9]*model.ug[9,t]) + bus2_seg1_t2[3] <= model.t02_5_t2[1,t]
model.bus02_5_l1_t2 = Constraint(model.PERIOD, rule = bus02_5_l1_t2)

def bus02_5_u1_t2(model,t):
    return bus2_seg1_t2[0]*100*model.Pg[9,t]*model.ug[9,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1_t2[2]* model.inertia_constant[9]*model.genD_Pmax[9]*model.ug[9,t]) + bus2_seg1_t2[3] + model.v02_5_t2[1,t]*A >= model.t02_5_t2[1,t]
model.bus02_5_u1_t2 = Constraint(model.PERIOD, rule = bus02_5_u1_t2)

def bus02_5_l2_t2(model,t):
    return bus2_seg2_t2[0]*100*model.Pg[9,t]*model.ug[9,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2_t2[2]* model.inertia_constant[9]*model.genD_Pmax[9]*model.ug[9,t]) + bus2_seg2_t2[3] <= model.t02_5_t2[1,t]
model.bus02_5_l2_t2 = Constraint(model.PERIOD,rule = bus02_5_l2_t2)

def bus02_5_u2_t2(model,t):
    return bus2_seg2_t2[0]*100*model.Pg[9,t]*model.ug[9,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2_t2[2]* model.inertia_constant[9]*model.genD_Pmax[9]*model.ug[9,t]) + bus2_seg2_t2[3] + (1 - model.v02_5_t2[1,t])*A >= model.t02_5_t2[1,t]
model.bus02_5_u2_t2 = Constraint(model.PERIOD,rule = bus02_5_u2_t2)

def bus02_5_t12_t2(model,t):
    return model.t02_5_t2[1,t] <= model.t02_5_t2[2,t]
model.bus02_5_t12_t2 = Constraint(model.PERIOD,rule = bus02_5_t12_t2)

def bus02_5_t21_t2(model,t):
    return model.t02_5_t2[2,t]<=model.t02_5_t2[1,t] + model.v02_5_t2[2,t]*A
model.bus02_5_t21_t2 = Constraint(model.PERIOD,rule = bus02_5_t21_t2)

def bus02_5_l3_t2(model,t):
    return bus2_seg3_t2[0]*100*model.Pg[9,t]*model.ug[9,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3_t2[2]* model.inertia_constant[9]*model.genD_Pmax[9]*model.ug[9,t]) + bus2_seg3_t2[3]  <= model.t02_5_t2[2,t]
model.bus02_5_l3_t2 = Constraint(model.PERIOD,rule = bus02_5_l3_t2)

def bus02_5_u3_t2(model,t):
    return bus2_seg3_t2[0]*100*model.Pg[9,t]*model.ug[9,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3_t2[2]* model.inertia_constant[9]*model.genD_Pmax[9]*model.ug[9,t]) + bus2_seg3_t2[3] + (1 - model.v02_5_t2[2,t])*A >= model.t02_5_t2[2,t]
model.bus02_5_u3_t2 = Constraint(model.PERIOD,rule = bus02_5_u3_t2)

def bus02_5_t23_t2(model,t):
    return model.t02_5_t2[2,t] <= model.t02_5_t2[3,t]
model.bus02_5_t23_t2 = Constraint(model.PERIOD,rule = bus02_5_t23_t2)

def bus02_5_t32_t2(model,t):
    return model.t02_5_t2[3,t]<=model.t02_5_t2[2,t] + model.v02_5_t2[3,t]*A
model.bus02_5_t32_t2 = Constraint(model.PERIOD,rule = bus02_5_t32_t2)

def bus02_5_l4_t2(model,t):
    return bus2_seg4_t2[0]*100*model.Pg[9,t]*model.ug[9,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4_t2[2]* model.inertia_constant[9]*model.genD_Pmax[9]*model.ug[9,t]) + bus2_seg4_t2[3] <= model.t02_5_t2[3,t]
model.bus02_5_l4_t2 = Constraint(model.PERIOD,rule = bus02_5_l4_t2)

def bus02_5_u4_t2(model,t):
    return bus2_seg4_t2[0]*100*model.Pg[9,t]*model.ug[9,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4_t2[2]* model.inertia_constant[9]*model.genD_Pmax[9]*model.ug[9,t]) + bus2_seg4_t2[3] + (1 - model.v02_5_t2[3,t])*A >= model.t02_5_t2[3,t]
model.bus02_5_u4_t2 = Constraint(model.PERIOD,rule = bus02_5_u4_t2)

def bus02_6_R_t2(model,t):
    return model.t02_6_t2[3,t] <= RoCoF
model.bus02_6_R_t2 = Constraint(model.PERIOD, rule = bus02_6_R_t2)

def bus02_6_l1_t2(model,t):
    return bus2_seg1_t2[0]*100*model.Pg[10,t]*model.ug[10,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1_t2[2]* model.inertia_constant[10]*model.genD_Pmax[10]*model.ug[10,t]) + bus2_seg1_t2[3] <= model.t02_6_t2[1,t]
model.bus02_6_l1_t2 = Constraint(model.PERIOD, rule = bus02_6_l1_t2)

def bus02_6_u1_t2(model,t):
    return bus2_seg1_t2[0]*100*model.Pg[10,t]*model.ug[10,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg1_t2[2]* model.inertia_constant[10]*model.genD_Pmax[10]*model.ug[10,t]) + bus2_seg1_t2[3] + model.v02_6_t2[1,t]*A >= model.t02_6_t2[1,t]
model.bus02_6_u1_t2 = Constraint(model.PERIOD, rule = bus02_6_u1_t2)

def bus02_6_l2_t2(model,t):
    return bus2_seg2_t2[0]*100*model.Pg[10,t]*model.ug[10,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2_t2[2]* model.inertia_constant[10]*model.genD_Pmax[10]*model.ug[10,t]) + bus2_seg2_t2[3] <= model.t02_6_t2[1,t]
model.bus02_6_l2_t2 = Constraint(model.PERIOD,rule = bus02_6_l2_t2)

def bus02_6_u2_t2(model,t):
    return bus2_seg2_t2[0]*100*model.Pg[10,t]*model.ug[10,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg2_t2[2]* model.inertia_constant[10]*model.genD_Pmax[10]*model.ug[10,t]) + bus2_seg2_t2[3] + (1 - model.v02_6_t2[1,t])*A >= model.t02_6_t2[1,t]
model.bus02_6_u2_t2 = Constraint(model.PERIOD,rule = bus02_6_u2_t2)

def bus02_6_t12_t2(model,t):
    return model.t02_6_t2[1,t] <= model.t02_6_t2[2,t]
model.bus02_6_t12_t2 = Constraint(model.PERIOD,rule = bus02_6_t12_t2)

def bus02_6_t21_t2(model,t):
    return model.t02_6_t2[2,t]<=model.t02_6_t2[1,t] + model.v02_6_t2[2,t]*A
model.bus02_6_t21_t2 = Constraint(model.PERIOD,rule = bus02_6_t21_t2)

def bus02_6_l3_t2(model,t):
    return bus2_seg3_t2[0]*100*model.Pg[10,t]*model.ug[10,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3_t2[2]* model.inertia_constant[10]*model.genD_Pmax[10]*model.ug[10,t]) + bus2_seg3_t2[3]  <= model.t02_6_t2[2,t]
model.bus02_6_l3_t2 = Constraint(model.PERIOD,rule = bus02_6_l3_t2)

def bus02_6_u3_t2(model,t):
    return bus2_seg3_t2[0]*100*model.Pg[10,t]*model.ug[10,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg3_t2[2]* model.inertia_constant[10]*model.genD_Pmax[10]*model.ug[10,t]) + bus2_seg3_t2[3] + (1 - model.v02_6_t2[2,t])*A >= model.t02_6_t2[2,t]
model.bus02_6_u3_t2 = Constraint(model.PERIOD,rule = bus02_6_u3_t2)

def bus02_6_t23_t2(model,t):
    return model.t02_6_t2[2,t] <= model.t02_6_t2[3,t]
model.bus02_6_t23_t2 = Constraint(model.PERIOD,rule = bus02_6_t23_t2)

def bus02_6_t32_t2(model,t):
    return model.t02_6_t2[3,t]<=model.t02_6_t2[2,t] + model.v02_6_t2[3,t]*A
model.bus02_6_t32_t2 = Constraint(model.PERIOD,rule = bus02_6_t32_t2)

def bus02_6_l4_t2(model,t):
    return bus2_seg4_t2[0]*100*model.Pg[10,t]*model.ug[10,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4_t2[2]* model.inertia_constant[10]*model.genD_Pmax[10]*model.ug[10,t]) + bus2_seg4_t2[3] <= model.t02_6_t2[3,t]
model.bus02_6_l4_t2 = Constraint(model.PERIOD,rule = bus02_6_l4_t2)

def bus02_6_u4_t2(model,t):
    return bus2_seg4_t2[0]*100*model.Pg[10,t]*model.ug[10,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus2_seg4_t2[2]* model.inertia_constant[10]*model.genD_Pmax[10]*model.ug[10,t]) + bus2_seg4_t2[3] + (1 - model.v02_6_t2[3,t])*A >= model.t02_6_t2[3,t]
model.bus02_6_u4_t2 = Constraint(model.PERIOD,rule = bus02_6_u4_t2)

# Node 07 locational RoCoF constraints

def bus07_1_R_t2(model,t):
    return model.t07_1_t2[3,t] <= RoCoF
model.bus07_1_R_t2 = Constraint(model.PERIOD, rule = bus07_1_R_t2)

def bus07_1_l1_t2(model,t):
    return bus7_seg1_t2[0]*100*model.Pg[11,t]*model.ug[11,t] + bus7_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg1_t2[2]* model.inertia_constant[11]*model.genD_Pmax[11]*model.ug[11,t]) + bus7_seg1_t2[3]<= model.t07_1_t2[1,t]
model.bus07_1_l1_t2 = Constraint(model.PERIOD, rule = bus07_1_l1_t2)

def bus07_1_u1_t2(model,t):
    return bus7_seg1_t2[0]*100*model.Pg[11,t]*model.ug[11,t] + bus7_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg1_t2[2]* model.inertia_constant[11]*model.genD_Pmax[11]*model.ug[11,t]) + bus7_seg1_t2[3] + model.v07_1_t2[1,t]*A >= model.t07_1_t2[1,t]
model.bus07_1_u1_t2 = Constraint(model.PERIOD, rule = bus07_1_u1_t2)

def bus07_1_l2_t2(model,t):
    return bus7_seg2_t2[0]*100*model.Pg[11,t]*model.ug[11,t] + bus7_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg2_t2[2]* model.inertia_constant[11]*model.genD_Pmax[11]*model.ug[11,t]) + bus7_seg2_t2[3] <= model.t07_1_t2[1,t]
model.bus07_1_l2_t2 = Constraint(model.PERIOD,rule = bus07_1_l2_t2)

def bus07_1_u2_t2(model,t):
    return bus7_seg2_t2[0]*100*model.Pg[11,t]*model.ug[11,t] + bus7_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg2_t2[2]* model.inertia_constant[11]*model.genD_Pmax[11]*model.ug[11,t]) + bus7_seg2_t2[3] + (1 - model.v07_1_t2[1,t])*A >= model.t07_1_t2[1,t]
model.bus07_1_u2_t2 = Constraint(model.PERIOD,rule = bus07_1_u2_t2)

def bus07_1_t12_t2(model,t):
    return model.t07_1_t2[1,t] <= model.t07_1_t2[2,t]
model.bus07_1_t12_t2 = Constraint(model.PERIOD,rule = bus07_1_t12_t2)

def bus07_1_t21_t2(model,t):
    return model.t07_1_t2[2,t]<=model.t07_1_t2[1,t] + model.v07_1_t2[2,t]*A
model.bus07_1_t21_t2 = Constraint(model.PERIOD,rule = bus07_1_t21_t2)

def bus07_1_l3_t2(model,t):
    return bus7_seg3_t2[0]*100*model.Pg[11,t]*model.ug[11,t] + bus7_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg3_t2[2]* model.inertia_constant[11]*model.genD_Pmax[11]*model.ug[11,t]) + bus7_seg3_t2[3] <= model.t07_1_t2[2,t]
model.bus07_1_l3_t2 = Constraint(model.PERIOD,rule = bus07_1_l3_t2)

def bus07_1_u3_t2(model,t):
    return bus7_seg3_t2[0]*100*model.Pg[11,t]*model.ug[11,t] + bus7_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg3_t2[2]* model.inertia_constant[11]*model.genD_Pmax[11]*model.ug[11,t]) + bus7_seg3_t2[3]+ (1 - model.v07_1_t2[2,t])*A >= model.t07_1_t2[2,t]
model.bus07_1_u3_t2 = Constraint(model.PERIOD,rule = bus07_1_u3_t2)

def bus07_1_t23_t2(model,t):
    return model.t07_1_t2[2,t] <= model.t07_1_t2[3,t]
model.bus07_1_t23_t2 = Constraint(model.PERIOD,rule = bus07_1_t23_t2)

def bus07_1_t32_t2(model,t):
    return model.t07_1_t2[3,t]<=model.t07_1_t2[2,t] + model.v07_1_t2[3,t]*A
model.bus07_1_t32_t2 = Constraint(model.PERIOD,rule = bus07_1_t32_t2)

def bus07_1_l4_t2(model,t):
    return bus7_seg4_t2[0]*100*model.Pg[11,t]*model.ug[11,t] + bus7_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg4_t2[2]* model.inertia_constant[11]*model.genD_Pmax[11]*model.ug[11,t]) + bus7_seg4_t2[3] <= model.t07_1_t2[3,t]
model.bus07_1_l4_t2 = Constraint(model.PERIOD,rule = bus07_1_l4_t2)

def bus07_1_u4_t2(model,t):
    return bus7_seg4_t2[0]*100*model.Pg[11,t]*model.ug[11,t] + bus7_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg4_t2[2]* model.inertia_constant[11]*model.genD_Pmax[11]*model.ug[11,t]) + bus7_seg4_t2[3] + (1 - model.v07_1_t2[3,t])*A >= model.t07_1_t2[3,t]
model.bus07_1_u4_t2 = Constraint(model.PERIOD,rule = bus07_1_u4_t2)

def bus07_2_R_t2(model,t):
    return model.t07_2_t2[3,t] <= RoCoF
model.bus07_2_R_t2 = Constraint(model.PERIOD, rule = bus07_2_R_t2)

def bus07_2_l1_t2(model,t):
    return bus7_seg1_t2[0]*100*model.Pg[12,t]*model.ug[12,t] + bus7_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg1_t2[2]* model.inertia_constant[12]*model.genD_Pmax[12]*model.ug[12,t]) + bus7_seg1_t2[3] <= model.t07_2_t2[1,t]
model.bus07_2_l1_t2 = Constraint(model.PERIOD, rule = bus07_2_l1_t2)

def bus07_2_u1_t2(model,t):
    return bus7_seg1_t2[0]*100*model.Pg[12,t]*model.ug[12,t] + bus7_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg1_t2[2]* model.inertia_constant[12]*model.genD_Pmax[12]*model.ug[12,t]) + bus7_seg1_t2[3]+ model.v07_2_t2[1,t]*A >= model.t07_2_t2[1,t]
model.bus07_2_u1_t2 = Constraint(model.PERIOD, rule = bus07_2_u1_t2)

def bus07_2_l2_t2(model,t):
    return bus7_seg2_t2[0]*100*model.Pg[12,t]*model.ug[12,t] + bus7_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg2_t2[2]* model.inertia_constant[12]*model.genD_Pmax[12]*model.ug[12,t]) + bus7_seg2_t2[3] <= model.t07_2_t2[1,t]
model.bus07_2_l2_t2 = Constraint(model.PERIOD,rule = bus07_2_l2_t2)

def bus07_2_u2_t2(model,t):
    return bus7_seg2_t2[0]*100*model.Pg[12,t]*model.ug[12,t] + bus7_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg2_t2[2]* model.inertia_constant[12]*model.genD_Pmax[12]*model.ug[12,t]) + bus7_seg2_t2[3] + (1 - model.v07_2_t2[1,t])*A >= model.t07_2_t2[1,t]
model.bus07_2_u2_t2 = Constraint(model.PERIOD,rule = bus07_2_u2_t2)

def bus07_2_t12_t2(model,t):
    return model.t07_2_t2[1,t] <= model.t07_2_t2[2,t]
model.bus07_2_t12_t2 = Constraint(model.PERIOD,rule = bus07_2_t12_t2)

def bus07_2_t21_t2(model,t):
    return model.t07_2_t2[2,t]<=model.t07_2_t2[1,t] + model.v07_2_t2[2,t]*A
model.bus07_2_t21_t2 = Constraint(model.PERIOD,rule = bus07_2_t21_t2)

def bus07_2_l3_t2(model,t):
    return bus7_seg3_t2[0]*100*model.Pg[12,t]*model.ug[12,t] + bus7_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg3_t2[2]* model.inertia_constant[12]*model.genD_Pmax[12]*model.ug[12,t]) + bus7_seg3_t2[3] <= model.t07_2_t2[2,t]
model.bus07_2_l3_t2 = Constraint(model.PERIOD,rule = bus07_2_l3_t2)

def bus07_2_u3_t2(model,t):
    return bus7_seg3_t2[0]*100*model.Pg[12,t]*model.ug[12,t] + bus7_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg3_t2[2]* model.inertia_constant[12]*model.genD_Pmax[12]*model.ug[12,t]) + bus7_seg3_t2[3] + (1 - model.v07_2_t2[2,t])*A >= model.t07_2_t2[2,t]
model.bus07_2_u3_t2 = Constraint(model.PERIOD,rule = bus07_2_u3_t2)

def bus07_2_t23_t2(model,t):
    return model.t07_2_t2[2,t] <= model.t07_2_t2[3,t]
model.bus07_2_t23_t2 = Constraint(model.PERIOD,rule = bus07_2_t23_t2)

def bus07_2_t32_t2(model,t):
    return model.t07_2_t2[3,t]<=model.t07_2_t2[2,t] + model.v07_2_t2[3,t]*A
model.bus07_2_t32_t2 = Constraint(model.PERIOD,rule = bus07_2_t32_t2)

def bus07_2_l4_t2(model,t):
    return bus7_seg4_t2[0]*100*model.Pg[12,t]*model.ug[12,t] + bus7_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg4_t2[2]* model.inertia_constant[12]*model.genD_Pmax[12]*model.ug[12,t]) + bus7_seg4_t2[3] <= model.t07_2_t2[3,t]
model.bus07_2_l4_t2 = Constraint(model.PERIOD,rule = bus07_2_l4_t2)

def bus07_2_u4_t2(model,t):
    return bus7_seg4_t2[0]*100*model.Pg[12,t]*model.ug[12,t] + bus7_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg4_t2[2]* model.inertia_constant[12]*model.genD_Pmax[12]*model.ug[12,t]) + bus7_seg4_t2[3] + (1 - model.v07_2_t2[3,t])*A >= model.t07_2_t2[3,t]
model.bus07_2_u4_t2 = Constraint(model.PERIOD,rule = bus07_2_u4_t2)

def bus07_3_R_t2(model,t):
    return model.t07_3_t2[3,t] <= RoCoF
model.bus07_3_R_t2 = Constraint(model.PERIOD, rule = bus07_3_R_t2)

def bus07_3_l1_t2(model,t):
    return bus7_seg1_t2[0]*100*model.Pg[13,t]*model.ug[13,t] + bus7_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg1_t2[2]* model.inertia_constant[13]*model.genD_Pmax[13]*model.ug[13,t]) + bus7_seg1_t2[3] <= model.t07_3_t2[1,t]
model.bus07_3_l1_t2 = Constraint(model.PERIOD, rule = bus07_3_l1_t2)

def bus07_3_u1_t2(model,t):
    return bus7_seg1_t2[0]*100*model.Pg[13,t]*model.ug[13,t] + bus7_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg1_t2[2]* model.inertia_constant[13]*model.genD_Pmax[13]*model.ug[13,t]) + bus7_seg1_t2[3] + model.v07_3_t2[1,t]*A >= model.t07_3_t2[1,t]
model.bus07_3_u1_t2 = Constraint(model.PERIOD, rule = bus07_3_u1_t2)

def bus07_3_l2_t2(model,t):
    return bus7_seg2_t2[0]*100*model.Pg[13,t]*model.ug[13,t] + bus7_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg2_t2[2]* model.inertia_constant[13]*model.genD_Pmax[13]*model.ug[13,t]) + bus7_seg2_t2[3] <= model.t07_3_t2[1,t]
model.bus07_3_l2_t2 = Constraint(model.PERIOD,rule = bus07_3_l2_t2)

def bus07_3_u2_t2(model,t):
    return bus7_seg2_t2[0]*100*model.Pg[13,t]*model.ug[13,t] + bus7_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg2_t2[2]* model.inertia_constant[13]*model.genD_Pmax[13]*model.ug[13,t]) + bus7_seg2_t2[3] + (1 - model.v07_3_t2[1,t])*A >= model.t07_3_t2[1,t]
model.bus07_3_u2_t2 = Constraint(model.PERIOD,rule = bus07_3_u2_t2)

def bus07_3_t12_t2(model,t):
    return model.t07_3_t2[1,t] <= model.t07_3_t2[2,t]
model.bus07_3_t12_t2 = Constraint(model.PERIOD,rule = bus07_3_t12_t2)

def bus07_3_t21_t2(model,t):
    return model.t07_3_t2[2,t]<=model.t07_3_t2[1,t] + model.v07_3_t2[2,t]*A
model.bus07_3_t21_t2 = Constraint(model.PERIOD,rule = bus07_3_t21_t2)

def bus07_3_l3_t2(model,t):
    return bus7_seg3_t2[0]*100*model.Pg[13,t]*model.ug[13,t] + bus7_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg3_t2[2]* model.inertia_constant[13]*model.genD_Pmax[13]*model.ug[13,t]) + bus7_seg3_t2[3]  <= model.t07_3_t2[2,t]
model.bus07_3_l3_t2 = Constraint(model.PERIOD,rule = bus07_3_l3_t2)

def bus07_3_u3_t2(model,t):
    return bus7_seg3_t2[0]*100*model.Pg[13,t]*model.ug[13,t] + bus7_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg3_t2[2]* model.inertia_constant[13]*model.genD_Pmax[13]*model.ug[13,t]) + bus7_seg3_t2[3]  + (1 - model.v07_3_t2[2,t])*A >= model.t07_3_t2[2,t]
model.bus07_3_u3_t2 = Constraint(model.PERIOD,rule = bus07_3_u3_t2)

def bus07_3_t23_t2(model,t):
    return model.t07_3_t2[2,t] <= model.t07_3_t2[3,t]
model.bus07_3_t23_t2 = Constraint(model.PERIOD,rule = bus07_3_t23_t2)

def bus07_3_t32_t2(model,t):
    return model.t07_3_t2[3,t]<=model.t07_3_t2[2,t] + model.v07_3_t2[3,t]*A
model.bus07_3_t32_t2 = Constraint(model.PERIOD,rule = bus07_3_t32_t2)

def bus07_3_l4_t2(model,t):
    return bus7_seg4_t2[0]*100*model.Pg[13,t]*model.ug[13,t] + bus7_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg4_t2[2]* model.inertia_constant[13]*model.genD_Pmax[13]*model.ug[13,t]) + bus7_seg4_t2[3] <= model.t07_3_t2[3,t]
model.bus07_3_l4_t2 = Constraint(model.PERIOD,rule = bus07_3_l4_t2)

def bus07_3_u4_t2(model,t):
    return bus7_seg4_t2[0]*100*model.Pg[13,t]*model.ug[13,t] + bus7_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus7_seg4_t2[2]* model.inertia_constant[13]*model.genD_Pmax[13]*model.ug[13,t]) + bus7_seg4_t2[3] + (1 - model.v07_3_t2[3,t])*A >= model.t07_3_t2[3,t]
model.bus07_3_u4_t2 = Constraint(model.PERIOD,rule = bus07_3_u4_t2)

# Node 13 locational RoCoF constraints

def bus13_1_R_t2(model,t):
    return model.t13_1_t2[3,t] <= RoCoF
model.bus13_1_R_t2 = Constraint(model.PERIOD, rule = bus13_1_R_t2)

def bus13_1_l1_t2(model,t):
    return bus13_seg1_t2[0]*100*model.Pg[14,t]*model.ug[14,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1_t2[2]* model.inertia_constant[14]*model.genD_Pmax[14]*model.ug[14,t]) + bus13_seg1_t2[3] <= model.t13_1_t2[1,t]
model.bus13_1_l1_t2 = Constraint(model.PERIOD, rule = bus13_1_l1_t2)

def bus13_1_u1_t2(model,t):
    return bus13_seg1_t2[0]*100*model.Pg[14,t]*model.ug[14,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1_t2[2]* model.inertia_constant[14]*model.genD_Pmax[14]*model.ug[14,t]) + bus13_seg1_t2[3] + model.v13_1_t2[1,t]*A >= model.t13_1_t2[1,t]
model.bus13_1_u1_t2 = Constraint(model.PERIOD, rule = bus13_1_u1_t2)

def bus13_1_l2_t2(model,t):
    return bus13_seg2_t2[0]*100*model.Pg[14,t]*model.ug[14,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2_t2[2]* model.inertia_constant[14]*model.genD_Pmax[14]*model.ug[14,t]) + bus13_seg2_t2[3] <= model.t13_1_t2[1,t]
model.bus13_1_l2_t2 = Constraint(model.PERIOD,rule = bus13_1_l2_t2)

def bus13_1_u2_t2(model,t):
    return bus13_seg2_t2[0]*100*model.Pg[14,t]*model.ug[14,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2_t2[2]* model.inertia_constant[14]*model.genD_Pmax[14]*model.ug[14,t]) + bus13_seg2_t2[3] + (1 - model.v13_1_t2[1,t])*A >= model.t13_1_t2[1,t]
model.bus13_1_u2_t2 = Constraint(model.PERIOD,rule = bus13_1_u2_t2)

def bus13_1_t12_t2(model,t):
    return model.t13_1_t2[1,t] <= model.t13_1_t2[2,t]
model.bus13_1_t12_t2 = Constraint(model.PERIOD,rule = bus13_1_t12_t2)

def bus13_1_t21_t2(model,t):
    return model.t13_1_t2[2,t]<=model.t13_1_t2[1,t] + model.v13_1_t2[2,t]*A
model.bus13_1_t21_t2 = Constraint(model.PERIOD,rule = bus13_1_t21_t2)

def bus13_1_l3_t2(model,t):
    return bus13_seg3_t2[0]*100*model.Pg[14,t]*model.ug[14,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3_t2[2]* model.inertia_constant[14]*model.genD_Pmax[14]*model.ug[14,t]) + bus13_seg3_t2[3] <= model.t13_1_t2[2,t]
model.bus13_1_l3_t2 = Constraint(model.PERIOD,rule = bus13_1_l3_t2)

def bus13_1_u3_t2(model,t):
    return bus13_seg3_t2[0]*100*model.Pg[14,t]*model.ug[14,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3_t2[2]* model.inertia_constant[14]*model.genD_Pmax[14]*model.ug[14,t]) + bus13_seg3_t2[3] + (1 - model.v13_1_t2[2,t])*A >= model.t13_1_t2[2,t]
model.bus13_1_u3_t2 = Constraint(model.PERIOD,rule = bus13_1_u3_t2)

def bus13_1_t23_t2(model,t):
    return model.t13_1_t2[2,t] <= model.t13_1_t2[3,t]
model.bus13_1_t23_t2 = Constraint(model.PERIOD,rule = bus13_1_t23_t2)

def bus13_1_t32_t2(model,t):
    return model.t13_1_t2[3,t]<=model.t13_1_t2[2,t] + model.v13_1_t2[3,t]*A
model.bus13_1_t32_t2 = Constraint(model.PERIOD,rule = bus13_1_t32_t2)

def bus13_1_l4_t2(model,t):
    return bus13_seg4_t2[0]*100*model.Pg[14,t]*model.ug[14,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4_t2[2]* model.inertia_constant[14]*model.genD_Pmax[14]*model.ug[14,t]) + bus13_seg4_t2[3] <= model.t13_1_t2[3,t]
model.bus13_1_l4_t2 = Constraint(model.PERIOD,rule = bus13_1_l4_t2)

def bus13_1_u4_t2(model,t):
    return bus13_seg4_t2[0]*100*model.Pg[14,t]*model.ug[14,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4_t2[2]* model.inertia_constant[14]*model.genD_Pmax[14]*model.ug[14,t]) + bus13_seg4_t2[3] + (1 - model.v13_1_t2[3,t])*A >= model.t13_1_t2[3,t]
model.bus13_1_u4_t2 = Constraint(model.PERIOD,rule = bus13_1_u4_t2)

def bus13_2_R_t2(model,t):
    return model.t13_2_t2[3,t] <= RoCoF
model.bus13_2_R_t2 = Constraint(model.PERIOD, rule = bus13_2_R_t2)

def bus13_2_l1_t2(model,t):
    return bus13_seg1_t2[0]*100*model.Pg[15,t]*model.ug[15,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1_t2[2]* model.inertia_constant[15]*model.genD_Pmax[15]*model.ug[15,t]) + bus13_seg1_t2[3] <= model.t13_2_t2[1,t]
model.bus13_2_l1_t2 = Constraint(model.PERIOD, rule = bus13_2_l1_t2)

def bus13_2_u1_t2(model,t):
    return bus13_seg1_t2[0]*100*model.Pg[15,t]*model.ug[15,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1_t2[2]* model.inertia_constant[15]*model.genD_Pmax[15]*model.ug[15,t]) + bus13_seg1_t2[3] + model.v13_2_t2[1,t]*A >= model.t13_2_t2[1,t]
model.bus13_2_u1_t2 = Constraint(model.PERIOD, rule = bus13_2_u1_t2)

def bus13_2_l2_t2(model,t):
    return bus13_seg2_t2[0]*100*model.Pg[15,t]*model.ug[15,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2_t2[2]* model.inertia_constant[15]*model.genD_Pmax[15]*model.ug[15,t]) + bus13_seg2_t2[3] <= model.t13_2_t2[1,t]
model.bus13_2_l2_t2 = Constraint(model.PERIOD,rule = bus13_2_l2_t2)

def bus13_2_u2_t2(model,t):
    return bus13_seg2_t2[0]*100*model.Pg[15,t]*model.ug[15,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2_t2[2]* model.inertia_constant[15]*model.genD_Pmax[15]*model.ug[15,t]) + bus13_seg2_t2[3]+ (1 - model.v13_2_t2[1,t])*A >= model.t13_2_t2[1,t]
model.bus13_2_u2_t2 = Constraint(model.PERIOD,rule = bus13_2_u2_t2)

def bus13_2_t12_t2(model,t):
    return model.t13_2_t2[1,t] <= model.t13_2_t2[2,t]
model.bus13_2_t12_t2 = Constraint(model.PERIOD,rule = bus13_2_t12_t2)

def bus13_2_t21_t2(model,t):
    return model.t13_2_t2[2,t]<=model.t13_2_t2[1,t] + model.v13_2_t2[2,t]*A
model.bus13_2_t21_t2 = Constraint(model.PERIOD,rule = bus13_2_t21_t2)

def bus13_2_l3_t2(model,t):
    return bus13_seg3_t2[0]*100*model.Pg[15,t]*model.ug[15,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3_t2[2]* model.inertia_constant[15]*model.genD_Pmax[15]*model.ug[15,t]) + bus13_seg3_t2[3]  <= model.t13_2_t2[2,t]
model.bus13_2_l3_t2 = Constraint(model.PERIOD,rule = bus13_2_l3_t2)

def bus13_2_u3_t2(model,t):
    return bus13_seg3_t2[0]*100*model.Pg[15,t]*model.ug[15,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3_t2[2]* model.inertia_constant[15]*model.genD_Pmax[15]*model.ug[15,t]) + bus13_seg3_t2[3] + (1 - model.v13_2_t2[2,t])*A >= model.t13_2_t2[2,t]
model.bus13_2_u3_t2 = Constraint(model.PERIOD,rule = bus13_2_u3_t2)

def bus13_2_t23_t2(model,t):
    return model.t13_2_t2[2,t] <= model.t13_2_t2[3,t]
model.bus13_2_t23_t2 = Constraint(model.PERIOD,rule = bus13_2_t23_t2)

def bus13_2_t32_t2(model,t):
    return model.t13_2_t2[3,t]<=model.t13_2_t2[2,t] + model.v13_2_t2[3,t]*A
model.bus13_2_t32_t2 = Constraint(model.PERIOD,rule = bus13_2_t32_t2)

def bus13_2_l4_t2(model,t):
    return bus13_seg4_t2[0]*100*model.Pg[15,t]*model.ug[15,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4_t2[2]* model.inertia_constant[15]*model.genD_Pmax[15]*model.ug[15,t]) + bus13_seg4_t2[3] <= model.t13_2_t2[3,t]
model.bus13_2_l4_t2 = Constraint(model.PERIOD,rule = bus13_2_l4_t2)

def bus13_2_u4_t2(model,t):
    return bus13_seg4_t2[0]*100*model.Pg[15,t]*model.ug[15,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4_t2[2]* model.inertia_constant[15]*model.genD_Pmax[15]*model.ug[15,t]) + bus13_seg4_t2[3] + (1 - model.v13_2_t2[3,t])*A >= model.t13_2_t2[3,t]
model.bus13_2_u4_t2 = Constraint(model.PERIOD,rule = bus13_2_u4_t2)

def bus13_3_R_t2(model,t):
    return model.t13_3_t2[3,t] <= RoCoF
model.bus13_3_R_t2 = Constraint(model.PERIOD, rule = bus13_3_R_t2)

def bus13_3_l1_t2(model,t):
    return bus13_seg1_t2[0]*100*model.Pg[16,t]*model.ug[16,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1_t2[2]* model.inertia_constant[16]*model.genD_Pmax[16]*model.ug[16,t]) + bus13_seg1_t2[3] <= model.t13_3_t2[1,t]
model.bus13_3_l1_t2 = Constraint(model.PERIOD, rule = bus13_3_l1_t2)

def bus13_3_u1_t2(model,t):
    return bus13_seg1_t2[0]*100*model.Pg[16,t]*model.ug[16,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1_t2[2]* model.inertia_constant[16]*model.genD_Pmax[16]*model.ug[16,t]) + bus13_seg1_t2[3] + model.v13_3_t2[1,t]*A >= model.t13_3_t2[1,t]
model.bus13_3_u1_t2 = Constraint(model.PERIOD, rule = bus13_3_u1_t2)

def bus13_3_l2_t2(model,t):
    return bus13_seg2_t2[0]*100*model.Pg[16,t]*model.ug[16,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2_t2[2]* model.inertia_constant[16]*model.genD_Pmax[16]*model.ug[16,t]) + bus13_seg2_t2[3] <= model.t13_3_t2[1,t]
model.bus13_3_l2_t2 = Constraint(model.PERIOD,rule = bus13_3_l2_t2)

def bus13_3_u2_t2(model,t):
    return bus13_seg2_t2[0]*100*model.Pg[16,t]*model.ug[16,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2_t2[2]* model.inertia_constant[16]*model.genD_Pmax[16]*model.ug[16,t]) + bus13_seg2_t2[3] + (1 - model.v13_3_t2[1,t])*A >= model.t13_3_t2[1,t]
model.bus13_3_u2_t2 = Constraint(model.PERIOD,rule = bus13_3_u2_t2)

def bus13_3_t12_t2(model,t):
    return model.t13_3_t2[1,t] <= model.t13_3_t2[2,t]
model.bus13_3_t12_t2 = Constraint(model.PERIOD,rule = bus13_3_t12_t2)

def bus13_3_t21_t2(model,t):
    return model.t13_3_t2[2,t]<=model.t13_3_t2[1,t] + model.v13_3_t2[2,t]*A
model.bus13_3_t21_t2 = Constraint(model.PERIOD,rule = bus13_3_t21_t2)

def bus13_3_l3_t2(model,t):
    return bus13_seg3_t2[0]*100*model.Pg[16,t]*model.ug[16,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3_t2[2]* model.inertia_constant[16]*model.genD_Pmax[16]*model.ug[16,t]) + bus13_seg3_t2[3] <= model.t13_3_t2[2,t]
model.bus13_3_l3_t2 = Constraint(model.PERIOD,rule = bus13_3_l3_t2)

def bus13_3_u3_t2(model,t):
    return bus13_seg3_t2[0]*100*model.Pg[16,t]*model.ug[16,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3_t2[2]* model.inertia_constant[16]*model.genD_Pmax[16]*model.ug[16,t]) + bus13_seg3_t2[3] + (1 - model.v13_3_t2[2,t])*A >= model.t13_3_t2[2,t]
model.bus13_3_u3_t2 = Constraint(model.PERIOD,rule = bus13_3_u3_t2)

def bus13_3_t23_t2(model,t):
    return model.t13_3_t2[2,t] <= model.t13_3_t2[3,t]
model.bus13_3_t23_t2 = Constraint(model.PERIOD,rule = bus13_3_t23_t2)

def bus13_3_t32_t2(model,t):
    return model.t13_3_t2[3,t]<=model.t13_3_t2[2,t] + model.v13_3_t2[3,t]*A
model.bus13_3_t32_t2 = Constraint(model.PERIOD,rule = bus13_3_t32_t2)

def bus13_3_l4_t2(model,t):
    return bus13_seg4_t2[0]*100*model.Pg[16,t]*model.ug[16,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4_t2[2]* model.inertia_constant[16]*model.genD_Pmax[16]*model.ug[16,t]) + bus13_seg4_t2[3] <= model.t13_3_t2[3,t]
model.bus13_3_l4_t2 = Constraint(model.PERIOD,rule = bus13_3_l4_t2)

def bus13_3_u4_t2(model,t):
    return bus13_seg4_t2[0]*100*model.Pg[16,t]*model.ug[16,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4_t2[2]* model.inertia_constant[16]*model.genD_Pmax[16]*model.ug[16,t]) + bus13_seg4_t2[3] + (1 - model.v13_3_t2[3,t])*A >= model.t13_3_t2[3,t]
model.bus13_3_u4_t2 = Constraint(model.PERIOD,rule = bus13_3_u4_t2)

def bus13_4_R_t2(model,t):
    return model.t13_4_t2[3,t] <= RoCoF
model.bus13_4_R_t2 = Constraint(model.PERIOD, rule = bus13_4_R_t2)

def bus13_4_l1_t2(model,t):
    return bus13_seg1_t2[0]*100*model.Pg[17,t]*model.ug[17,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1_t2[2]* model.inertia_constant[17]*model.genD_Pmax[17]*model.ug[17,t]) + bus13_seg1_t2[3] <= model.t13_4_t2[1,t]
model.bus13_4_l1_t2 = Constraint(model.PERIOD, rule = bus13_4_l1_t2)

def bus13_4_u1_t2(model,t):
    return bus13_seg1_t2[0]*100*model.Pg[17,t]*model.ug[17,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1_t2[2]* model.inertia_constant[17]*model.genD_Pmax[17]*model.ug[17,t]) + bus13_seg1_t2[3] + model.v13_4_t2[1,t]*A >= model.t13_4_t2[1,t]
model.bus13_4_u1_t2 = Constraint(model.PERIOD, rule = bus13_4_u1_t2)

def bus13_4_l2_t2(model,t):
    return bus13_seg2_t2[0]*100*model.Pg[17,t]*model.ug[17,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2_t2[2]* model.inertia_constant[17]*model.genD_Pmax[17]*model.ug[17,t]) + bus13_seg2_t2[3] <= model.t13_4_t2[1,t]
model.bus13_4_l2_t2 = Constraint(model.PERIOD,rule = bus13_4_l2_t2)

def bus13_4_u2_t2(model,t):
    return bus13_seg2_t2[0]*100*model.Pg[17,t]*model.ug[17,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2_t2[2]* model.inertia_constant[17]*model.genD_Pmax[17]*model.ug[17,t]) + bus13_seg2_t2[3]+ (1 - model.v13_4_t2[1,t])*A >= model.t13_4_t2[1,t]
model.bus13_4_u2_t2 = Constraint(model.PERIOD,rule = bus13_4_u2_t2)

def bus13_4_t12_t2(model,t):
    return model.t13_4_t2[1,t] <= model.t13_4_t2[2,t]
model.bus13_4_t12_t2 = Constraint(model.PERIOD,rule = bus13_4_t12_t2)

def bus13_4_t21_t2(model,t):
    return model.t13_4_t2[2,t]<=model.t13_4_t2[1,t] + model.v13_4_t2[2,t]*A
model.bus13_4_t21_t2 = Constraint(model.PERIOD,rule = bus13_4_t21_t2)

def bus13_4_l3_t2(model,t):
    return bus13_seg3_t2[0]*100*model.Pg[17,t]*model.ug[17,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3_t2[2]* model.inertia_constant[17]*model.genD_Pmax[17]*model.ug[17,t]) + bus13_seg3_t2[3]  <= model.t13_4_t2[2,t]
model.bus13_4_l3_t2 = Constraint(model.PERIOD,rule = bus13_4_l3_t2)

def bus13_4_u3_t2(model,t):
    return bus13_seg3_t2[0]*100*model.Pg[17,t]*model.ug[17,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3_t2[2]* model.inertia_constant[17]*model.genD_Pmax[17]*model.ug[17,t]) + bus13_seg3_t2[3] + (1 - model.v13_4_t2[2,t])*A >= model.t13_4_t2[2,t]
model.bus13_4_u3_t2 = Constraint(model.PERIOD,rule = bus13_4_u3_t2)

def bus13_4_t23_t2(model,t):
    return model.t13_4_t2[2,t] <= model.t13_4_t2[3,t]
model.bus13_4_t23_t2 = Constraint(model.PERIOD,rule = bus13_4_t23_t2)

def bus13_4_t32_t2(model,t):
    return model.t13_4_t2[3,t]<=model.t13_4_t2[2,t] + model.v13_4_t2[3,t]*A
model.bus13_4_t32_t2 = Constraint(model.PERIOD,rule = bus13_4_t32_t2)

def bus13_4_l4_t2(model,t):
    return bus13_seg4_t2[0]*100*model.Pg[17,t]*model.ug[17,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4_t2[2]* model.inertia_constant[17]*model.genD_Pmax[17]*model.ug[17,t]) + bus13_seg4_t2[3] <= model.t13_4_t2[3,t]
model.bus13_4_l4_t2 = Constraint(model.PERIOD,rule = bus13_4_l4_t2)

def bus13_4_u4_t2(model,t):
    return bus13_seg4_t2[0]*100*model.Pg[17,t]*model.ug[17,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4_t2[2]* model.inertia_constant[17]*model.genD_Pmax[17]*model.ug[17,t]) + bus13_seg4_t2[3] + (1 - model.v13_4_t2[3,t])*A >= model.t13_4_t2[3,t]
model.bus13_4_u4_t2 = Constraint(model.PERIOD,rule = bus13_4_u4_t2)

def bus13_5_R_t2(model,t):
    return model.t13_5_t2[3,t] <= RoCoF
model.bus13_5_R_t2 = Constraint(model.PERIOD, rule = bus13_5_R_t2)

def bus13_5_l1_t2(model,t):
    return bus13_seg1_t2[0]*100*model.Pg[18,t]*model.ug[18,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1_t2[2]* model.inertia_constant[18]*model.genD_Pmax[18]*model.ug[18,t]) + bus13_seg1_t2[3] <= model.t13_5_t2[1,t]
model.bus13_5_l1_t2 = Constraint(model.PERIOD, rule = bus13_5_l1_t2)

def bus13_5_u1_t2(model,t):
    return bus13_seg1_t2[0]*100*model.Pg[18,t]*model.ug[18,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1_t2[2]* model.inertia_constant[18]*model.genD_Pmax[18]*model.ug[18,t]) + bus13_seg1_t2[3] + model.v13_5_t2[1,t]*A >= model.t13_5_t2[1,t]
model.bus13_5_u1_t2 = Constraint(model.PERIOD, rule = bus13_5_u1_t2)

def bus13_5_l2_t2(model,t):
    return bus13_seg2_t2[0]*100*model.Pg[18,t]*model.ug[18,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2_t2[2]* model.inertia_constant[18]*model.genD_Pmax[18]*model.ug[18,t]) + bus13_seg2_t2[3] <= model.t13_5_t2[1,t]
model.bus13_5_l2_t2 = Constraint(model.PERIOD,rule = bus13_5_l2_t2)

def bus13_5_u2_t2(model,t):
    return bus13_seg2_t2[0]*100*model.Pg[18,t]*model.ug[18,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2_t2[2]* model.inertia_constant[18]*model.genD_Pmax[18]*model.ug[18,t]) + bus13_seg2_t2[3] + (1 - model.v13_5_t2[1,t])*A >= model.t13_5_t2[1,t]
model.bus13_5_u2_t2 = Constraint(model.PERIOD,rule = bus13_5_u2_t2)

def bus13_5_t12_t2(model,t):
    return model.t13_5_t2[1,t] <= model.t13_5_t2[2,t]
model.bus13_5_t12_t2 = Constraint(model.PERIOD,rule = bus13_5_t12_t2)

def bus13_5_t21_t2(model,t):
    return model.t13_5_t2[2,t]<=model.t13_5_t2[1,t] + model.v13_5_t2[2,t]*A
model.bus13_5_t21_t2 = Constraint(model.PERIOD,rule = bus13_5_t21_t2)

def bus13_5_l3_t2(model,t):
    return bus13_seg3_t2[0]*100*model.Pg[18,t]*model.ug[18,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3_t2[2]* model.inertia_constant[18]*model.genD_Pmax[18]*model.ug[18,t]) + bus13_seg3_t2[3] <= model.t13_5_t2[2,t]
model.bus13_5_l3_t2 = Constraint(model.PERIOD,rule = bus13_5_l3_t2)

def bus13_5_u3_t2(model,t):
    return bus13_seg3_t2[0]*100*model.Pg[18,t]*model.ug[18,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3_t2[2]* model.inertia_constant[18]*model.genD_Pmax[18]*model.ug[18,t]) + bus13_seg3_t2[3] + (1 - model.v13_5_t2[2,t])*A >= model.t13_5_t2[2,t]
model.bus13_5_u3_t2 = Constraint(model.PERIOD,rule = bus13_5_u3_t2)

def bus13_5_t23_t2(model,t):
    return model.t13_5_t2[2,t] <= model.t13_5_t2[3,t]
model.bus13_5_t23_t2 = Constraint(model.PERIOD,rule = bus13_5_t23_t2)

def bus13_5_t32_t2(model,t):
    return model.t13_5_t2[3,t]<=model.t13_5_t2[2,t] + model.v13_5_t2[3,t]*A
model.bus13_5_t32_t2 = Constraint(model.PERIOD,rule = bus13_5_t32_t2)

def bus13_5_l4_t2(model,t):
    return bus13_seg4_t2[0]*100*model.Pg[18,t]*model.ug[18,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4_t2[2]* model.inertia_constant[18]*model.genD_Pmax[18]*model.ug[18,t]) + bus13_seg4_t2[3] <= model.t13_5_t2[3,t]
model.bus13_5_l4_t2 = Constraint(model.PERIOD,rule = bus13_5_l4_t2)

def bus13_5_u4_t2(model,t):
    return bus13_seg4_t2[0]*100*model.Pg[18,t]*model.ug[18,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4_t2[2]* model.inertia_constant[18]*model.genD_Pmax[18]*model.ug[18,t]) + bus13_seg4_t2[3] + (1 - model.v13_5_t2[3,t])*A >= model.t13_5_t2[3,t]
model.bus13_5_u4_t2 = Constraint(model.PERIOD,rule = bus13_5_u4_t2)

def bus13_6_R_t2(model,t):
    return model.t13_6_t2[3,t] <= RoCoF
model.bus13_6_R_t2 = Constraint(model.PERIOD, rule = bus13_6_R_t2)

def bus13_6_l1_t2(model,t):
    return bus13_seg1_t2[0]*100*model.Pg[19,t]*model.ug[19,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1_t2[2]* model.inertia_constant[19]*model.genD_Pmax[19]*model.ug[19,t]) + bus13_seg1_t2[3] <= model.t13_6_t2[1,t]
model.bus13_6_l1_t2 = Constraint(model.PERIOD, rule = bus13_6_l1_t2)

def bus13_6_u1_t2(model,t):
    return bus13_seg1_t2[0]*100*model.Pg[19,t]*model.ug[19,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg1_t2[2]* model.inertia_constant[19]*model.genD_Pmax[19]*model.ug[19,t]) + bus13_seg1_t2[3] + model.v13_6_t2[1,t]*A >= model.t13_6_t2[1,t]
model.bus13_6_u1_t2 = Constraint(model.PERIOD, rule = bus13_6_u1_t2)

def bus13_6_l2_t2(model,t):
    return bus13_seg2_t2[0]*100*model.Pg[19,t]*model.ug[19,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2_t2[2]* model.inertia_constant[19]*model.genD_Pmax[19]*model.ug[19,t]) + bus13_seg2_t2[3] <= model.t13_6_t2[1,t]
model.bus13_6_l2_t2 = Constraint(model.PERIOD,rule = bus13_6_l2_t2)

def bus13_6_u2_t2(model,t):
    return bus13_seg2_t2[0]*100*model.Pg[19,t]*model.ug[19,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg2_t2[2]* model.inertia_constant[19]*model.genD_Pmax[19]*model.ug[19,t]) + bus13_seg2_t2[3] + (1 - model.v13_6_t2[1,t])*A >= model.t13_6_t2[1,t]
model.bus13_6_u2_t2 = Constraint(model.PERIOD,rule = bus13_6_u2_t2)

def bus13_6_t12_t2(model,t):
    return model.t13_6_t2[1,t] <= model.t13_6_t2[2,t]
model.bus13_6_t12_t2 = Constraint(model.PERIOD,rule = bus13_6_t12_t2)

def bus13_6_t21_t2(model,t):
    return model.t13_6_t2[2,t]<=model.t13_6_t2[1,t] + model.v13_6_t2[2,t]*A
model.bus13_6_t21_t2 = Constraint(model.PERIOD,rule = bus13_6_t21_t2)

def bus13_6_l3_t2(model,t):
    return bus13_seg3_t2[0]*100*model.Pg[19,t]*model.ug[19,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3_t2[2]* model.inertia_constant[19]*model.genD_Pmax[19]*model.ug[19,t]) + bus13_seg3_t2[3] <= model.t13_6_t2[2,t]
model.bus13_6_l3_t2 = Constraint(model.PERIOD,rule = bus13_6_l3_t2)

def bus13_6_u3_t2(model,t):
    return bus13_seg3_t2[0]*100*model.Pg[19,t]*model.ug[19,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg3_t2[2]* model.inertia_constant[19]*model.genD_Pmax[19]*model.ug[19,t]) + bus13_seg3_t2[3] + (1 - model.v13_6_t2[2,t])*A >= model.t13_6_t2[2,t]
model.bus13_6_u3_t2 = Constraint(model.PERIOD,rule = bus13_6_u3_t2)

def bus13_6_t23_t2(model,t):
    return model.t13_6_t2[2,t] <= model.t13_6_t2[3,t]
model.bus13_6_t23_t2 = Constraint(model.PERIOD,rule = bus13_6_t23_t2)

def bus13_6_t32_t2(model,t):
    return model.t13_6_t2[3,t]<=model.t13_6_t2[2,t] + model.v13_6_t2[3,t]*A
model.bus13_6_t32_t2 = Constraint(model.PERIOD,rule = bus13_6_t32_t2)

def bus13_6_l4_t2(model,t):
    return bus13_seg4_t2[0]*100*model.Pg[19,t]*model.ug[19,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4_t2[2]* model.inertia_constant[19]*model.genD_Pmax[19]*model.ug[19,t]) + bus13_seg4_t2[3] <= model.t13_6_t2[3,t]
model.bus13_6_l4_t2 = Constraint(model.PERIOD,rule = bus13_6_l4_t2)

def bus13_6_u4_t2(model,t):
    return bus13_seg4_t2[0]*100*model.Pg[19,t]*model.ug[19,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus13_seg4_t2[2]* model.inertia_constant[19]*model.genD_Pmax[19]*model.ug[19,t]) + bus13_seg4_t2[3] + (1 - model.v13_6_t2[3,t])*A >= model.t13_6_t2[3,t]
model.bus13_6_u4_t2 = Constraint(model.PERIOD,rule = bus13_6_u4_t2)

# Node 15 locational RoCoF constraints

def bus15_R_t2(model,g,t):
    if g >= 20:
        if g <= 24:
            return model.t15_t2[3,t] <= RoCoF
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_R_t2 = Constraint(model.GEND, model.PERIOD,rule = bus15_R_t2)

def bus15_l1_t2(model,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg1_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg1_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg1_t2[3]<= model.t15_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_l1_t2 = Constraint(model.GEND,model.PERIOD,rule = bus15_l1_t2)

def bus15_u1_t2(model,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg1_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg1_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg1_t2[3] + model.v15_t2[1,t]*A >= model.t15_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_u1_t2 = Constraint(model.GEND, model.PERIOD, rule = bus15_u1_t2)

def bus15_l2_t2(model,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg2_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg2_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg2_t2[3] <= model.t15_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_l2_t2 = Constraint(model.GEND, model.PERIOD,rule = bus15_l2_t2)

def bus15_u2_t2(model,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg2_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg2_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg2_t2[3] + ( 1 - model.v15_t2[1, t]) * A >= model.t15_t2[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_u2_t2 = Constraint(model.GEND, model.PERIOD, rule = bus15_u2_t2)

def bus15_t12_t2(model,g,t):
    if g >= 20:
        if g <= 24:
            return model.t15_t2[1,t] <= model.t15_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_t12_t2 = Constraint(model.GEND, model.PERIOD,rule = bus15_t12_t2)

def bus15_t21_t2(model,g,t):
    if g >= 20:
        if g <= 24:
            return model.t15_t2[2,t]<=model.t15_t2[1,t] + model.v15_t2[2,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_t21_t2 = Constraint(model.GEND,model.PERIOD, rule = bus15_t21_t2)

def bus15_l3_t2(model,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg3_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg3_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg3_t2[3] <= model.t15_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_l3_t2 = Constraint(model.GEND,model.PERIOD, rule = bus15_l3_t2)

def bus15_u3_t2(model,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg3_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg3_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg3_t2[3] + (1 - model.v15_t2[2,t])*A >= model.t15_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_u3_t2 = Constraint(model.GEND, model.PERIOD,rule = bus15_u3_t2)

def bus15_t23_t2(model,g,t):
    if g >= 20:
        if g <= 24:
            return model.t15_t2[2,t] <= model.t15_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_t23_t2 = Constraint(model.GEND,model.PERIOD, rule = bus15_t23_t2)

def bus15_t32_t2(model,g,t):
    if g >= 20:
        if g <= 24:
            return model.t15_t2[3,t]<=model.t15_t2[2,t] + model.v15_t2[3,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_t32_t2 = Constraint(model.GEND, model.PERIOD,rule = bus15_t32_t2)

def bus15_l4_t2(model,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg4_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg4_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg4_t2[3] <= model.t15_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_l4_t2 = Constraint(model.GEND,model.PERIOD,rule = bus15_l4_t2)

def bus15_u4_t2(model,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg4_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg4_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg4_t2[3] + (1 - model.v15_t2[3,t])*A >= model.t15_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_u4_t2 = Constraint(model.GEND,model.PERIOD,rule = bus15_u4_t2)

# Node 15_1 constraint
def bus15_1_R_t2(model,g,t):
    if g >= 25:
        if g <= 26:
            return model.t15_1_t2[3,t] <= RoCoF
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_R_t2 = Constraint(model.GEND, model.PERIOD,rule = bus15_1_R_t2)

def bus15_1_l1_t2(model,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg1_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg1_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg1_t2[3] <= model.t15_1_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_l1_t2 = Constraint(model.GEND,model.PERIOD,rule = bus15_1_l1_t2)

def bus15_1_u1_t2(model,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg1_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg1_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg1_t2[3]+ model.v15_1_t2[1,t]*A >= model.t15_1_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_u1_t2 = Constraint(model.GEND, model.PERIOD, rule = bus15_1_u1_t2)

def bus15_1_l2_t2(model,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg2_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg2_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg2_t2[3] <= model.t15_1_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_l2_t2 = Constraint(model.GEND, model.PERIOD,rule = bus15_1_l2_t2)

def bus15_1_u2_t2(model,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg2_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg2_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg2_t2[3] + ( 1 - model.v15_1_t2[1, t]) * A >= model.t15_1_t2[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_u2_t2 = Constraint(model.GEND, model.PERIOD, rule = bus15_1_u2_t2)

def bus15_1_t12_t2(model,g,t):
    if g >= 25:
        if g <= 26:
            return model.t15_1_t2[1,t] <= model.t15_1_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_t12_t2 = Constraint(model.GEND, model.PERIOD,rule = bus15_1_t12_t2)

def bus15_1_t21_t2(model,g,t):
    if g >= 25:
        if g <= 26:
            return model.t15_1_t2[2,t]<=model.t15_1_t2[1,t] + model.v15_1_t2[2,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_t21_t2 = Constraint(model.GEND,model.PERIOD, rule = bus15_1_t21_t2)

def bus15_1_l3_t2(model,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg3_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg3_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg3_t2[3] <= model.t15_1_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_l3_t2 = Constraint(model.GEND,model.PERIOD, rule = bus15_1_l3_t2)

def bus15_1_u3_t2(model,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg3_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg3_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg3_t2[3] + (1 - model.v15_1_t2[2,t])*A >= model.t15_1_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_u3_t2 = Constraint(model.GEND, model.PERIOD,rule = bus15_1_u3_t2)

def bus15_1_t23_t2(model,g,t):
    if g >= 25:
        if g <= 26:
            return model.t15_1_t2[2,t] <= model.t15_1_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_t23_t2 = Constraint(model.GEND,model.PERIOD, rule = bus15_1_t23_t2)

def bus15_1_t32_t2(model,g,t):
    if g >= 25:
        if g <= 26:
            return model.t15_1_t2[3,t]<=model.t15_1_t2[2,t] + model.v15_1_t2[3,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_t32_t2 = Constraint(model.GEND, model.PERIOD,rule = bus15_1_t32_t2)

def bus15_1_l4_t2(model,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg4_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg4_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg4_t2[3] <= model.t15_1_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_l4_t2 = Constraint(model.GEND,model.PERIOD,rule = bus15_1_l4_t2)

def bus15_1_u4_t2(model,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg4_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus15_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus15_seg4_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus15_seg4_t2[3] + (1 - model.v15_1_t2[3,t])*A >= model.t15_1_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus15_1_u4_t2 = Constraint(model.GEND,model.PERIOD,rule = bus15_1_u4_t2)

# Node 16 locational RoCoF constraints

def bus16_R_t2(model,g,t):
    if g >= 27:
        if g <= 29:
            return model.t16_t2[3,t] <= RoCoF
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_R_t2 = Constraint(model.GEND, model.PERIOD,rule = bus16_R_t2)

def bus16_l1_t2(model,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg1_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus16_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus16_seg1_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus16_seg1_t2[3] <= model.t16_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_l1_t2 = Constraint(model.GEND,model.PERIOD,rule = bus16_l1_t2)

def bus16_u1_t2(model,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg1_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus16_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus16_seg1_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus16_seg1_t2[3]+ model.v16_t2[1,t]*A >= model.t16_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_u1_t2 = Constraint(model.GEND, model.PERIOD, rule = bus16_u1_t2)

def bus16_l2_t2(model,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg2_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus16_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus16_seg2_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus16_seg2_t2[3]<= model.t16_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_l2_t2 = Constraint(model.GEND, model.PERIOD,rule = bus16_l2_t2)

def bus16_u2_t2(model,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg2_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus16_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus16_seg2_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus16_seg2_t2[3] + ( 1 - model.v16_t2[1, t]) * A >= model.t16_t2[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_u2_t2 = Constraint(model.GEND, model.PERIOD, rule = bus16_u2_t2)

def bus16_t12_t2(model,g,t):
    if g >= 27:
        if g <= 29:
            return model.t16_t2[1,t] <= model.t16_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_t12_t2 = Constraint(model.GEND, model.PERIOD,rule = bus16_t12_t2)

def bus16_t21_t2(model,g,t):
    if g >= 27:
        if g <= 29:
            return model.t16_t2[2,t]<=model.t16_t2[1,t] + model.v16[2,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_t21_t2 = Constraint(model.GEND,model.PERIOD, rule = bus16_t21_t2)

def bus16_l3_t2(model,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg3_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus16_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus16_seg3_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus16_seg3_t2[3] <= model.t16_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_l3_t2 = Constraint(model.GEND,model.PERIOD, rule = bus16_l3_t2)

def bus16_u3_t2(model,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg3_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus16_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus16_seg3_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus16_seg3_t2[3] + (1 - model.v16_t2[2,t])*A >= model.t16_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_u3_t2 = Constraint(model.GEND, model.PERIOD,rule = bus16_u3_t2)

def bus16_t23_t2(model,g,t):
    if g >= 27:
        if g <= 29:
            return model.t16_t2[2,t] <= model.t16_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_t23_t2 = Constraint(model.GEND,model.PERIOD, rule = bus16_t23_t2)

def bus16_t32_t2(model,g,t):
    if g >= 27:
        if g <= 29:
            return model.t16_t2[3,t]<=model.t16_t2[2,t] + model.v16_t2[3,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_t32_t2 = Constraint(model.GEND, model.PERIOD,rule = bus16_t32_t2)

def bus16_l4_t2(model,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg4_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus16_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus16_seg4_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus16_seg4_t2[3] <= model.t16_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_l4_t2 = Constraint(model.GEND,model.PERIOD,rule = bus16_l4_t2)

def bus16_u4_t2(model,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg4_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus16_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus16_seg4_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus16_seg4_t2[3] + (1 - model.v16_t2[3,t])*A >= model.t16_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus16_u4_t2 = Constraint(model.GEND,model.PERIOD,rule = bus16_u4_t2)

# Node 18 locational RoCoF constraints

def bus18_R_t2(model,t):
    return model.t18_t2[3,t] <= RoCoF
model.bus18_R_t2 = Constraint(model.PERIOD, rule = bus18_R_t2)

def bus18_l1_t2(model,t):
    return bus18_seg1_t2[0]*100*model.Pg[30,t]*model.ug[30,t] + bus18_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus18_seg1_t2[2]* model.inertia_constant[30]*model.genD_Pmax[30]*model.ug[30,t]) + bus18_seg1_t2[3] <= model.t18_t2[1,t]
model.bus18_l1_t2 = Constraint(model.PERIOD, rule = bus18_l1_t2)

def bus18_u1_t2(model,t):
    return bus18_seg1_t2[0]*100*model.Pg[30,t]*model.ug[30,t] + bus18_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus18_seg1_t2[2]* model.inertia_constant[30]*model.genD_Pmax[30]*model.ug[30,t]) + bus18_seg1_t2[3] + model.v18_t2[1,t]*A >= model.t18_t2[1,t]
model.bus18_u1_t2 = Constraint(model.PERIOD, rule = bus18_u1_t2)

def bus18_l2_t2(model,t):
    return bus18_seg2_t2[0]*100*model.Pg[30,t]*model.ug[30,t] + bus18_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus18_seg2_t2[2]* model.inertia_constant[30]*model.genD_Pmax[30]*model.ug[30,t]) + bus18_seg2_t2[3] <= model.t18_t2[1,t]
model.bus18_l2_t2 = Constraint(model.PERIOD,rule = bus18_l2_t2)

def bus18_u2_t2(model,t):
    return bus18_seg2_t2[0]*100*model.Pg[30,t]*model.ug[30,t] + bus18_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus18_seg2_t2[2]* model.inertia_constant[30]*model.genD_Pmax[30]*model.ug[30,t]) + bus18_seg2_t2[3]+ (1 - model.v18_t2[1,t])*A >= model.t18_t2[1,t]
model.bus18_u2_t2 = Constraint(model.PERIOD,rule = bus18_u2_t2)

def bus18_t12_t2(model,t):
    return model.t18_t2[1,t] <= model.t18_t2[2,t]
model.bus18_t12_t2 = Constraint(model.PERIOD,rule = bus18_t12_t2)

def bus18_t21_t2(model,t):
    return model.t18_t2[2,t]<=model.t18_t2[1,t] + model.v18_t2[2,t]*A
model.bus18_t21_t2 = Constraint(model.PERIOD,rule = bus18_t21_t2)

def bus18_l3_t2(model,t):
    return bus18_seg3_t2[0]*100*model.Pg[30,t]*model.ug[30,t] + bus18_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus18_seg3_t2[2]* model.inertia_constant[30]*model.genD_Pmax[30]*model.ug[30,t]) + bus18_seg3_t2[3]<= model.t18_t2[2,t]
model.bus18_l3_t2 = Constraint(model.PERIOD,rule = bus18_l3_t2)

def bus18_u3_t2(model,t):
    return bus18_seg3_t2[0]*100*model.Pg[30,t]*model.ug[30,t] + bus18_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus18_seg3_t2[2]* model.inertia_constant[30]*model.genD_Pmax[30]*model.ug[30,t]) + bus18_seg3_t2[3] + (1 - model.v18_t2[2,t])*A >= model.t18_t2[2,t]
model.bus18_u3_t2 = Constraint(model.PERIOD,rule = bus18_u3_t2)

def bus18_t23_t2(model,t):
    return model.t18_t2[2,t] <= model.t18_t2[3,t]
model.bus18_t23_t2 = Constraint(model.PERIOD,rule = bus18_t23_t2)

def bus18_t32_t2(model,t):
    return model.t18_t2[3,t]<=model.t18_t2[2,t] + model.v18_t2[3,t]*A
model.bus18_t32_t2 = Constraint(model.PERIOD,rule = bus18_t32_t2)

def bus18_l4_t2(model,t):
    return bus18_seg4_t2[0]*100*model.Pg[30,t]*model.ug[30,t] + bus18_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus18_seg4_t2[2]* model.inertia_constant[30]*model.genD_Pmax[30]*model.ug[30,t]) + bus18_seg4_t2[3] <= model.t18_t2[3,t]
model.bus18_l4_t2 = Constraint(model.PERIOD,rule = bus18_l4_t2)

def bus18_u4_t2(model,t):
    return bus18_seg4_t2[0]*100*model.Pg[30,t]*model.ug[30,t] + bus18_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus18_seg4_t2[2]* model.inertia_constant[30]*model.genD_Pmax[30]*model.ug[30,t]) + bus18_seg4_t2[3] + (1 - model.v18_t2[3,t])*A >= model.t18_t2[3,t]
model.bus18_u4_t2 = Constraint(model.PERIOD,rule = bus18_u4_t2)

# Node 21 locational RoCoF constraints

def bus21_R_t2(model,g,t):
    if g >= 31:
        if g <= 35:
            return model.t21_t2[3,t] <= RoCoF
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_R_t2 = Constraint(model.GEND, model.PERIOD,rule = bus21_R_t2)

def bus21_l1_t2(model,g,t):
    if g >= 31:
        if g <= 35:
            return bus21_seg1_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus21_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus21_seg1_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus21_seg1_t2[3] <= model.t21_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_l1_t2 = Constraint(model.GEND,model.PERIOD,rule = bus21_l1_t2)

def bus21_u1_t2(model,g,t):
    if g >= 31:
        if g <= 35:
            return bus21_seg1_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus21_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus21_seg1_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus21_seg1_t2[3] + model.v21_t2[1,t]*A >= model.t21_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_u1_t2 = Constraint(model.GEND, model.PERIOD, rule = bus21_u1_t2)

def bus21_l2_t2(model,g,t):
    if g >= 31:
        if g <= 35:
            return bus21_seg2_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus21_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus21_seg2_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus21_seg2_t2[3] <= model.t21_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_l2_t2 = Constraint(model.GEND, model.PERIOD,rule = bus21_l2_t2)

def bus21_u2_t2(model,g,t):
    if g >= 31:
        if g <= 35:
            return bus21_seg2_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus21_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus21_seg2_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus21_seg2_t2[3] + ( 1 - model.v21_t2[1, t]) * A >= model.t21_t2[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_u2_t2 = Constraint(model.GEND, model.PERIOD, rule = bus21_u2_t2)

def bus21_t12_t2(model,g,t):
    if g >= 31:
        if g <= 35:
            return model.t21_t2[1,t] <= model.t21_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_t12_t2 = Constraint(model.GEND, model.PERIOD,rule = bus21_t12_t2)

def bus21_t21_t2(model,g,t):
    if g >= 31:
        if g <= 35:
            return model.t21_t2[2,t]<=model.t21_t2[1,t] + model.v21_t2[2,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_t21_t2 = Constraint(model.GEND,model.PERIOD, rule = bus21_t21_t2)

def bus21_l3_t2(model,g,t):
    if g >= 31:
        if g <= 35:
            return bus21_seg3_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus21_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus21_seg3_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus21_seg3_t2[3] <= model.t21_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_l3_t2 = Constraint(model.GEND,model.PERIOD, rule = bus21_l3_t2)

def bus21_u3_t2(model,g,t):
    if g >= 31:
        if g <= 35:
            return bus21_seg3_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus21_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus21_seg3_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus21_seg3_t2[3] + (1 - model.v21_t2[2,t])*A >= model.t21_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_u3_t2 = Constraint(model.GEND, model.PERIOD,rule = bus21_u3_t2)

def bus21_t23_t2(model,g,t):
    if g >= 31:
        if g <= 35:
            return model.t21_t2[2,t] <= model.t21_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_t23_t2 = Constraint(model.GEND,model.PERIOD, rule = bus21_t23_t2)

def bus21_t32_t2(model,g,t):
    if g >= 31:
        if g <= 35:
            return model.t21_t2[3,t]<=model.t21_t2[2,t] + model.v21_t2[3,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_t32_t2 = Constraint(model.GEND, model.PERIOD,rule = bus21_t32_t2)

def bus21_l4_t2(model,g,t):
    if g >= 31:
        if g <= 35:
            return bus21_seg4_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus21_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus21_seg4_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus21_seg4_t2[3] <= model.t21_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_l4_t2 = Constraint(model.GEND,model.PERIOD,rule = bus21_l4_t2)

def bus21_u4_t2(model,g,t):
    if g >= 31:
        if g <= 35:
            return bus21_seg4_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus21_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus21_seg4_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus21_seg4_t2[3] + (1 - model.v21_t2[3,t])*A >= model.t21_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus21_u4_t2 = Constraint(model.GEND,model.PERIOD,rule = bus21_u4_t2)

# Node 22 constraints
def bus22_R_t2(model,t):
    return model.t22_t2[3,t] <= RoCoF
model.bus22_R_t2 = Constraint(model.PERIOD, rule = bus22_R_t2)

def bus22_l1_t2(model,t):
    return bus22_seg1_t2[0]*100*model.Pg[36,t]*model.ug[36,t] + bus22_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus22_seg1_t2[2]* model.inertia_constant[36]*model.genD_Pmax[36]*model.ug[36,t]) + bus22_seg1_t2[3] <= model.t22_t2[1,t]
model.bus22_l1_t2 = Constraint(model.PERIOD, rule = bus22_l1_t2)

def bus22_u1_t2(model,t):
    return bus22_seg1_t2[0]*100*model.Pg[36,t]*model.ug[36,t] + bus22_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus22_seg1_t2[2]* model.inertia_constant[36]*model.genD_Pmax[36]*model.ug[36,t]) + bus22_seg1_t2[3] + model.v22_t2[1,t]*A >= model.t22_t2[1,t]
model.bus22_u1_t2 = Constraint(model.PERIOD, rule = bus22_u1_t2)

def bus22_l2_t2(model,t):
    return bus22_seg2_t2[0]*100*model.Pg[36,t]*model.ug[36,t] + bus22_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus22_seg2_t2[2]* model.inertia_constant[36]*model.genD_Pmax[36]*model.ug[36,t]) + bus22_seg2_t2[3] <= model.t22_t2[1,t]
model.bus22_l2_t2 = Constraint(model.PERIOD,rule = bus22_l2_t2)

def bus22_u2_t2(model,t):
    return bus22_seg2_t2[0]*100*model.Pg[36,t]*model.ug[36,t] + bus22_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus22_seg2_t2[2]* model.inertia_constant[36]*model.genD_Pmax[36]*model.ug[36,t]) + bus22_seg2_t2[3] + (1 - model.v22_t2[1,t])*A >= model.t22_t2[1,t]
model.bus22_u2_t2 = Constraint(model.PERIOD,rule = bus22_u2_t2)

def bus22_t12_t2(model,t):
    return model.t22_t2[1,t] <= model.t22_t2[2,t]
model.bus22_t12_t2 = Constraint(model.PERIOD,rule = bus22_t12_t2)

def bus22_t21_t2(model,t):
    return model.t22_t2[2,t]<=model.t22_t2[1,t] + model.v22_t2[2,t]*A
model.bus22_t21_t2 = Constraint(model.PERIOD,rule = bus22_t21_t2)

def bus22_l3_t2(model,t):
    return bus22_seg3_t2[0]*100*model.Pg[36,t]*model.ug[36,t] + bus22_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus22_seg3_t2[2]* model.inertia_constant[36]*model.genD_Pmax[36]*model.ug[36,t]) + bus22_seg3_t2[3] <= model.t22_t2[2,t]
model.bus22_l3_t2 = Constraint(model.PERIOD,rule = bus22_l3_t2)

def bus22_u3_t2(model,t):
    return bus22_seg3_t2[0]*100*model.Pg[36,t]*model.ug[36,t] + bus22_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus22_seg3_t2[2]* model.inertia_constant[36]*model.genD_Pmax[36]*model.ug[36,t]) + bus22_seg3_t2[3] + (1 - model.v22_t2[2,t])*A >= model.t22_t2[2,t]
model.bus22_u3_t2 = Constraint(model.PERIOD,rule = bus22_u3_t2)

def bus22_t23_t2(model,t):
    return model.t22_t2[2,t] <= model.t22_t2[3,t]
model.bus22_t23_t2 = Constraint(model.PERIOD,rule = bus22_t23_t2)

def bus22_t32_t2(model,t):
    return model.t22_t2[3,t]<=model.t22_t2[2,t] + model.v22_t2[3,t]*A
model.bus22_t32_t2 = Constraint(model.PERIOD,rule = bus22_t32_t2)

def bus22_l4_t2(model,t):
    return bus22_seg4_t2[0]*100*model.Pg[36,t]*model.ug[36,t] + bus22_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus22_seg4_t2[2]* model.inertia_constant[36]*model.genD_Pmax[36]*model.ug[36,t]) + bus22_seg4_t2[3] <= model.t22_t2[3,t]
model.bus22_l4_t2 = Constraint(model.PERIOD,rule = bus22_l4_t2)

def bus22_u4_t2(model,t):
    return bus22_seg4_t2[0]*100*model.Pg[36,t]*model.ug[36,t] + bus22_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus22_seg4_t2[2]* model.inertia_constant[36]*model.genD_Pmax[36]*model.ug[36,t]) + bus22_seg4_t2[3] + (1 - model.v22_t2[3,t])*A >= model.t22_t2[3,t]
model.bus22_u4_t2 = Constraint(model.PERIOD,rule = bus22_u4_t2)


# Node 23 locational RoCoF constraints

def bus23_R_t2(model,g,t):
    if g >= 39:
        if g <= 41:
            return model.t23_t2[3,t] <= RoCoF
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_R_t2 = Constraint(model.GEND, model.PERIOD,rule = bus23_R_t2)

def bus23_l1_t2(model,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg1_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus23_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus23_seg1_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus23_seg1_t2[3]  <= model.t23_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_l1_t2 = Constraint(model.GEND,model.PERIOD,rule = bus23_l1_t2)

def bus23_u1_t2(model,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg1_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus23_seg1_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus23_seg1_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus23_seg1_t2[3]  + model.v23_t2[1,t]*A >= model.t23_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_u1_t2 = Constraint(model.GEND, model.PERIOD, rule = bus23_u1_t2)

def bus23_l2_t2(model,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg2_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus23_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus23_seg2_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus23_seg2_t2[3]  <= model.t23_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_l2_t2 = Constraint(model.GEND, model.PERIOD,rule = bus23_l2_t2)

def bus23_u2_t2(model,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg2_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus23_seg2_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus23_seg2_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus23_seg2_t2[3]  + ( 1 - model.v23_t2[1, t]) * A >= model.t23_t2[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_u2_t2 = Constraint(model.GEND, model.PERIOD, rule = bus23_u2_t2)

def bus23_t12_t2(model,g,t):
    if g >= 39:
        if g <= 41:
            return model.t23_t2[1,t] <= model.t23_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_t12_t2 = Constraint(model.GEND, model.PERIOD,rule = bus23_t12_t2)

def bus23_t21_t2(model,g,t):
    if g >= 39:
        if g <= 41:
            return model.t23_t2[2,t]<=model.t23_t2[1,t] + model.v23_t2[2,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_t21_t2 = Constraint(model.GEND,model.PERIOD, rule = bus23_t21_t2)

def bus23_l3_t2(model,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg3_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus23_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus23_seg3_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus23_seg3_t2[3]  <= model.t23_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_l3_t2 = Constraint(model.GEND,model.PERIOD, rule = bus23_l3_t2)

def bus23_u3_t2(model,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg3_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus23_seg3_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus23_seg3_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus23_seg3_t2[3]  + (1 - model.v23_t2[2,t])*A >= model.t23_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_u3_t2 = Constraint(model.GEND, model.PERIOD,rule = bus23_u3_t2)

def bus23_t23_t2(model,g,t):
    if g >= 39:
        if g <= 41:
            return model.t23_t2[2,t] <= model.t23_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_t23_t2 = Constraint(model.GEND,model.PERIOD, rule = bus23_t23_t2)

def bus23_t32_t2(model,g,t):
    if g >= 39:
        if g <= 41:
            return model.t23_t2[3,t]<=model.t23_t2[2,t] + model.v23_t2[3,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_t32_t2 = Constraint(model.GEND, model.PERIOD,rule = bus23_t32_t2)

def bus23_l4_t2(model,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg4_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus23_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus23_seg4_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus23_seg4_t2[3]  <= model.t23_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_l4_t2 = Constraint(model.GEND,model.PERIOD,rule = bus23_l4_t2)

def bus23_u4_t2(model,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg4_t2[0]*100*model.Pg[g,t]*model.ug[g,t] + bus23_seg4_t2[1]/(fn*np.pi)*(sum(model.inertia_constant[j]*model.genD_Pmax[j]*model.ug[j,t] for j in model.GEND) + bus23_seg4_t2[2]*model.inertia_constant[g]*model.genD_Pmax[g]*model.ug[g,t]) +bus23_seg4_t2[3]  + (1 - model.v23_t2[3,t])*A >= model.t23_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
model.bus23_u4_t2 = Constraint(model.GEND,model.PERIOD,rule = bus23_u4_t2)


# Virtual Inertia/ sync inertia compensation
#def Virtual_Inertia(model,j,t):
#   return model.Vi[j,t]>=0
#model.virtualcon = Constraint(model.BUS,model.PERIOD,rule=Virtual_Inertia)

instance = model.create_instance('./dataFile24BusAllinertia41sen_4EP.dat')
for t in range(23):
    instance.theta[13,t+1].fixed = True
    instance.theta[13,t+1].value = 0
SCUCsolver = SolverFactory('gurobi')
SCUCsolver.options.mipgap = 0.0
results = SCUCsolver.solve(instance)
print("\nresults.Solution.Status: " + str(results.Solution.Status))
print("\nresults.solver.status: " + str(results.solver.status))
print("\nresults.solver.termination_condition: " + str(results.solver.termination_condition))
print("\nresults.solver.termination_message: " + str(results.solver.termination_message))
print('\nminimize cost: ' + str(instance.obj()))

'''
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

writer_1 = pd.ExcelWriter('./SEN60_PWL_Data_EP_4.xlsx')
genunit.to_excel(writer_1, index=False,encoding='utf-8',sheet_name='Sheet')
writer_1.save()
writer_2 = pd.ExcelWriter('./SEN60_PWL_UC_EP_4.xlsx')
Data.to_excel(writer_2, index=False,encoding='utf-8',sheet_name='Sheet')
writer_2.save()
writer_3 = pd.ExcelWriter('./SEN60_PWL_RE_EP_4.xlsx')
re.to_excel(writer_3, index=False,encoding='utf-8',sheet_name='Sheet')
writer_3.save()
'''



m = AbstractModel()
# Sets
m.BUS = Set()
m.GEND = Set()
m.BRANCH = Set()
m.PERIOD = Set()

# Bus Paramters
m.bus_num = Param(m.BUS)
m.bus_Pd = Param(m.BUS)
m.bus_Solar = Param(m.BUS)
m.bus_Wind = Param(m.BUS)
# Generators Parameters
m.genD_bus = Param(m.GEND)
m.genD_minUP = Param(m.GEND)
m.genD_minDN = Param(m.GEND)
m.genD_status = Param(m.GEND)
m.genD_Pmax = Param(m.GEND)
m.genD_Pmin = Param(m.GEND)
m.genC_Startup = Param(m.GEND)
m.genC_Cost = Param(m.GEND)
m.genC_NLoad = Param(m.GEND)
m.SPRamp = Param(m.GEND)
m.NSRamp = Param(m.GEND)
m.HRamp = Param(m.GEND)
m.StartRamp = Param(m.GEND)
m.gen_Style = Param(m.GEND)
m.inertia_constant = Param(m.GEND)
m.resC = Param(m.GEND)

# Branch Parameters
m.branch_fbus = Param(m.BRANCH)
m.branch_tbus = Param(m.BRANCH)
m.branch_b = Param(m.BRANCH)
m.branch_rateA = Param(m.BRANCH)
m.branch_rateC = Param(m.BRANCH)
m.branch_radial = Param(m.BRANCH)

# Load
m.load_pcnt = Param(m.PERIOD)
m.Wind_pcnt = Param(m.PERIOD)
m.Solar_pcnt = Param(m.PERIOD)
# Variables
m.ug = Param(m.GEND, m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.vg = Param(m.GEND, m.PERIOD, mutable=True, within = Binary, initialize= 0)
m.theta = Var(m.BUS,m.PERIOD)
m.Pg = Var(m.GEND,m.PERIOD)
m.Pk = Var(m.BRANCH,m.PERIOD)
m.Vi = Var(m.BUS,m.PERIOD)
m.kg = Var(m.GEND, m.PERIOD)
m.re = Var(m.GEND,m.PERIOD)
BaseMVA = 100

# PWL of nodal
m.SEG = Set()
m.t01 = Var(m.SEG,m.PERIOD)
m.v01 = Param(m.SEG, m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t02_1 = Var(m.SEG,m.PERIOD)
m.v02_1 = Param(m.SEG, m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t02_2 = Var(m.SEG,m.PERIOD)
m.v02_2 = Param(m.SEG, m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t02_3 = Var(m.SEG,m.PERIOD)
m.v02_3 = Param(m.SEG, m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t02_4 = Var(m.SEG,m.PERIOD)
m.v02_4 = Param(m.SEG, m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t02_5 = Var(m.SEG,m.PERIOD)
m.v02_5 = Param(m.SEG, m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t02_6 = Var(m.SEG,m.PERIOD)
m.v02_6 = Param(m.SEG, m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t07_1 = Var(m.SEG,m.PERIOD)
m.v07_1 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t07_2 = Var(m.SEG,m.PERIOD)
m.v07_2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t07_3 = Var(m.SEG,m.PERIOD)
m.v07_3 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t13_1 = Var(m.SEG,m.PERIOD)
m.v13_1 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t13_2 = Var(m.SEG,m.PERIOD)
m.v13_2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t13_3 = Var(m.SEG,m.PERIOD)
m.v13_3 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t13_4 = Var(m.SEG,m.PERIOD)
m.v13_4 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t13_5 = Var(m.SEG,m.PERIOD)
m.v13_5 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t13_6 = Var(m.SEG,m.PERIOD)
m.v13_6 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t15 = Var(m.SEG,m.PERIOD)
m.v15 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t15_1 = Var(m.SEG,m.PERIOD)
m.v15_1 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t16 = Var(m.SEG,m.PERIOD)
m.v16 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t18 = Var(m.SEG,m.PERIOD)
m.v18 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t21 = Var(m.SEG,m.PERIOD)
m.v21 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.num = Param(m.SEG)
m.t22 = Var(m.SEG,m.PERIOD)
m.v22 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t23 = Var(m.SEG,m.PERIOD)
m.v23 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)

m.t01_t2 = Var(m.SEG,m.PERIOD)
m.v01_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t02_1_t2 = Var(m.SEG,m.PERIOD)
m.v02_1_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t02_2_t2 = Var(m.SEG,m.PERIOD)
m.v02_2_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t02_3_t2 = Var(m.SEG,m.PERIOD)
m.v02_3_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t02_4_t2 = Var(m.SEG,m.PERIOD)
m.v02_4_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t02_5_t2 = Var(m.SEG,m.PERIOD)
m.v02_5_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t02_6_t2 = Var(m.SEG,m.PERIOD)
m.v02_6_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t07_1_t2 = Var(m.SEG,m.PERIOD)
m.v07_1_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t07_2_t2 = Var(m.SEG,m.PERIOD)
m.v07_2_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t07_3_t2 = Var(m.SEG,m.PERIOD)
m.v07_3_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t13_1_t2 = Var(m.SEG,m.PERIOD)
m.v13_1_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t13_2_t2 = Var(m.SEG,m.PERIOD)
m.v13_2_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t13_3_t2 = Var(m.SEG,m.PERIOD)
m.v13_3_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t13_4_t2 = Var(m.SEG,m.PERIOD)
m.v13_4_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t13_5_t2 = Var(m.SEG,m.PERIOD)
m.v13_5_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t13_6_t2 = Var(m.SEG,m.PERIOD)
m.v13_6_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t15_t2 = Var(m.SEG,m.PERIOD)
m.v15_t2 =Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t15_1_t2 = Var(m.SEG,m.PERIOD)
m.v15_1_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t16_t2 = Var(m.SEG,m.PERIOD)
m.v16_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t18_t2 = Var(m.SEG,m.PERIOD)
m.v18_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t21_t2 = Var(m.SEG,m.PERIOD)
m.v21_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t22_t2 = Var(m.SEG,m.PERIOD)
m.v22_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)
m.t23_t2 = Var(m.SEG,m.PERIOD)
m.v23_t2 = Param(m.SEG,m.PERIOD, mutable=True, within = Binary,initialize= 0)

A =1000

# Objective Function
def obj_cost(m):
    return sum(m.genC_Cost[g]*m.Pg[g,t]*m.ug[g,t]*BaseMVA +  m.genC_Startup[g]*m.vg[g,t] +  m.resC[g]*m.re[g,t]*m.ug[g,t]*BaseMVA for g in m.GEND for t in m.PERIOD) #
    #     + sum(m.Vi[j,t]*virtual_price for j in m.BUS for t in m.PERIOD)
m.obj = Objective(rule=obj_cost,sense=minimize)

# Nodal balance constraints
def power_balance(m,bus,t):
    return -sum(m.Pk[j,t] for j in m.BRANCH if m.branch_fbus[j] == bus) + sum(m.Pk[j,t] for j in m.BRANCH if m.branch_tbus[j] == bus) == \
           -sum(m.Pg[g,t]*m.ug[g,t] for g in m.GEND if m.genD_bus[g] == bus) + m.bus_Pd[bus]*m.load_pcnt[t]/(100*BaseMVA) \
           -sen* m.bus_Wind[bus]*m.Wind_pcnt[t]/100 - r0* m.bus_Solar[bus]*m.Solar_pcnt[t]/100
m.power_balance = Constraint(m.BUS, m.PERIOD, rule=power_balance)

def gen_limit_min(m,j,t):
    return m.Pg[j,t] >= m.genD_Pmin[j]*m.ug[j,t]/BaseMVA
m.gen_min = Constraint(m.GEND,m.PERIOD,rule=gen_limit_min)

def gen_limit_max(m,j,t):
    return m.Pg[j,t]+ m.re[j,t] <= m.genD_Pmax[j]*m.ug[j,t]/BaseMVA
m.gen_max = Constraint(m.GEND,m.PERIOD,rule=gen_limit_max)

def reserve_limit(m,j,t):
    return 0<=m.re[j,t]<=m.HRamp[j]/BaseMVA
m.reserve = Constraint(m.GEND,m.PERIOD,rule = reserve_limit)

def reserve_sum(m,g,t):
    return sum(m.re[j,t] for j in m.GEND)>=m.Pg[g,t]+ m.re[g,t]
m.sum_re = Constraint(m.GEND,m.PERIOD,rule=reserve_sum)

def gen_ramping(m,g,t):
    if t >= 2:
        expr = m.Pg[g,t] - m.Pg[g,t-1] - m.HRamp[g]/BaseMVA
        return expr <= 0
    else:
        return Constraint.Skip
m.gen_ramping = Constraint(m.GEND,m.PERIOD,rule=gen_ramping)

def gen_ramping2(m,g,t):
    if t >= 2:
        expr = m.Pg[g,t-1] - m.Pg[g,t] - m.HRamp[g]/BaseMVA
        return expr <= 0
    else:
        return Constraint.Skip
m.gen_ramping2 = Constraint(m.GEND,m.PERIOD,rule=gen_ramping2)

# Line flow
def line_flow(m,j,t):
    return m.Pk[j,t] == -(m.theta[m.branch_fbus[j],t]-m.theta[m.branch_tbus[j],t])*m.branch_b[j]
m.line_flow = Constraint(m.BRANCH,m.PERIOD,rule=line_flow)

# Line thermal constraints
def line_flow_limitslow(m,j,t):
    return -m.branch_rateA[j] <= BaseMVA*m.Pk[j,t]
m.line_flow_limitslow = Constraint(m.BRANCH,m.PERIOD,rule=line_flow_limitslow)

def line_flow_limitshigh(m,j,t):
    return BaseMVA*m.Pk[j,t] <= m.branch_rateA[j]
m.line_flow_limitshigh = Constraint(m.BRANCH,m.PERIOD,rule=line_flow_limitshigh)

# Unit satus  constraints
def genUV(m,g,t):
    if t>=2:
        expr = m.ug[g,t] - m.ug[g,t-1] - m.vg[g,t]
        return expr <= 0
    else:
        return Constraint.Skip
m.gen_UV = Constraint(m.GEND,m.PERIOD,rule=genUV)

# Initialize constraint
def geninitial(m,g):
    expr = m.ug[g,1] - m.vg[g,1]
    return expr <= 0
m.Cons_initial = Constraint(m.GEND, rule = geninitial )

# PWL nodal RoCoF Constraints


# Node 01 locational RoCoF constraints

def bus01_R(m,t):
    return m.t01[3,t] <= RoCoF
m.bus01_R = Constraint(m.PERIOD,rule = bus01_R)

def bus01_l1(m,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg1[0]*100*m.Pg[g,t]*m.ug[g,t]+ bus1_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND)) + bus1_seg1[2] * m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t] + bus1_seg1[3] <= m.t01[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus01_l1 = Constraint(m.GEND,m.PERIOD,rule = bus01_l1)

def bus01_u1(m,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg1[0]*100*m.Pg[g,t]*m.ug[g,t] + bus1_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus1_seg1[2] * m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) + bus1_seg1[3] + m.v01[1,t]*A >= m.t01[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus01_u1 = Constraint(m.GEND, m.PERIOD, rule = bus01_u1)

def bus01_l2(m,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus1_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND)) + bus1_seg2[2] * m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t] +  bus1_seg2[3] <= m.t01[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus01_l2 = Constraint(m.GEND, m.PERIOD,rule = bus01_l2)

def bus01_u2(m,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg2[0]*100*m.Pg[g,t]*m.ug[g,t] +  bus1_seg2[1]/(fn*np.pi)*(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND)) + bus1_seg2[2] * m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]+  bus1_seg2[3] + ( 1 - m.v01[1, t]) * A >= m.t01[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus01_u2 = Constraint(m.GEND, m.PERIOD, rule = bus01_u2)

def bus01_t12(m,t):
    return m.t01[1,t] <= m.t01[2,t]
m.bus01_t12 = Constraint( m.PERIOD,rule = bus01_t12)

def bus01_t21(m,t):
    return m.t01[2,t]<=m.t01[1,t] + m.v01[2,t]*A
m.bus01_t21 = Constraint(m.PERIOD, rule = bus01_t21)

def bus01_l3(m,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg3[0]*100*m.Pg[g,t]*m.ug[g,t] + bus1_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus1_seg3[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus1_seg3[3]<= m.t01[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus01_l3 = Constraint(m.GEND,m.PERIOD, rule = bus01_l3)

def bus01_u3(m,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg3[0]*100*m.Pg[g,t]*m.ug[g,t] + bus1_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND)) + bus1_seg3[2]* m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]  + bus1_seg3[3] + (1 - m.v01[2,t])*A >= m.t01[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus01_u3 = Constraint(m.GEND, m.PERIOD,rule = bus01_u3)

def bus01_t23(m,t):
    return m.t01[2,t] <= m.t01[3,t]
m.bus01_t23 = Constraint(m.PERIOD, rule = bus01_t23)

def bus01_t32(m,t):
    return m.t01[3,t]<=m.t01[2,t] + m.v01[3,t]*A
m.bus01_t32 = Constraint(m.PERIOD,rule = bus01_t32)

def bus01_l4(m,g,t):
    if g >= 1:
        if g <= 4:
             return bus1_seg4[0]*m.Pg[g,t]*m.ug[g,t] + bus1_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND)) + bus1_seg4[2]* m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t] + bus1_seg4[3] <= m.t01[3,t]
        else:
             return Constraint.Skip
    else:
        return Constraint.Skip
m.bus01_l4 = Constraint(m.GEND,m.PERIOD,rule = bus01_l4)

def bus01_u4(m,g,t):
    if g >= 1:
        if g <= 4:
             return bus1_seg4[0]*m.Pg[g,t]*m.ug[g,t] + bus1_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND)) + bus1_seg4[2]* m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t] + bus1_seg4[3] +(1 - m.v01[3,t])*A >= m.t01[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus01_u4 = Constraint(m.GEND,m.PERIOD,rule = bus01_u4)

# Node 02 locational RoCoF constraints

def bus02_1_R(m,t):
    return m.t02_1[3,t] <= RoCoF
m.bus02_1_R = Constraint(m.PERIOD, rule = bus02_1_R)

def bus02_1_l1(m,t):
    return bus2_seg1[0]*100*m.Pg[5,t]*m.ug[5,t] + bus2_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1[2] * m.inertia_constant[5]*m.genD_Pmax[5]*m.ug[5,t]) + bus2_seg1[3] <= m.t02_1[1,t]
m.bus02_1_l1 = Constraint(m.PERIOD, rule = bus02_1_l1)

def bus02_1_u1(m,t):
    return bus2_seg1[0]*100*m.Pg[5,t]*m.ug[5,t] + bus2_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1[2] * m.inertia_constant[5]*m.genD_Pmax[5]*m.ug[5,t]) + bus2_seg1[3] + m.v02_1[1,t]*A >= m.t02_1[1,t]
m.bus02_1_u1 = Constraint(m.PERIOD, rule = bus02_1_u1)

def bus02_1_l2(m,t):
    return bus2_seg2[0]*100*m.Pg[5,t]*m.ug[5,t] + bus2_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2[2] * m.inertia_constant[5]*m.genD_Pmax[5]*m.ug[5,t]) + bus2_seg2[3] <= m.t02_1[1,t]
m.bus02_1_l2 = Constraint(m.PERIOD,rule = bus02_1_l2)

def bus02_1_u2(m,t):
    return bus2_seg2[0]*100*m.Pg[5,t]*m.ug[5,t] + bus2_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2[2] * m.inertia_constant[5]*m.genD_Pmax[5]*m.ug[5,t]) + bus2_seg2[3] + (1 - m.v02_1[1,t])*A >= m.t02_1[1,t]
m.bus02_1_u2 = Constraint(m.PERIOD,rule = bus02_1_u2)

def bus02_1_t12(m,t):
    return m.t02_1[1,t] <= m.t02_1[2,t]
m.bus02_1_t12 = Constraint(m.PERIOD,rule = bus02_1_t12)

def bus02_1_t21(m,t):
    return m.t02_1[2,t]<=m.t02_1[1,t] + m.v02_1[2,t]*A
m.bus02_1_t21 = Constraint(m.PERIOD,rule = bus02_1_t21)

def bus02_1_l3(m,t):
    return bus2_seg3[0]*100*m.Pg[5,t]*m.ug[5,t] + bus2_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3[2]* m.inertia_constant[5]*m.genD_Pmax[5]*m.ug[5,t]) + bus2_seg3[3]  <= m.t02_1[2,t]
m.bus02_1_l3 = Constraint(m.PERIOD,rule = bus02_1_l3)

def bus02_1_u3(m,t):
    return bus2_seg3[0]*100*m.Pg[5,t]*m.ug[5,t] + bus2_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3[2]* m.inertia_constant[5]*m.genD_Pmax[5]*m.ug[5,t]) + bus2_seg3[3]   + (1 - m.v02_1[2,t])*A >= m.t02_1[2,t]
m.bus02_1_u3 = Constraint(m.PERIOD,rule = bus02_1_u3)

def bus02_1_t23(m,t):
    return m.t02_1[2,t] <= m.t02_1[3,t]
m.bus02_1_t23 = Constraint(m.PERIOD,rule = bus02_1_t23)

def bus02_1_t32(m,t):
    return m.t02_1[3,t]<=m.t02_1[2,t] + m.v02_1[3,t]*A
m.bus02_1_t32 = Constraint(m.PERIOD,rule = bus02_1_t32)

def bus02_1_l4(m,t):
    return bus2_seg4[0]*100*m.Pg[5,t]*m.ug[5,t] + bus2_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4[2]* m.inertia_constant[5]*m.genD_Pmax[5]*m.ug[5,t]) + bus2_seg4[3]  <= m.t02_1[3,t]
m.bus02_1_l4 = Constraint(m.PERIOD,rule = bus02_1_l4)

def bus02_1_u4(m,t):
    return bus2_seg4[0]*100*m.Pg[5,t]*m.ug[5,t] + bus2_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4[2]* m.inertia_constant[5]*m.genD_Pmax[5]*m.ug[5,t]) + bus2_seg4[3]+(1 - m.v02_1[3,t])*A >= m.t02_1[3,t]
m.bus02_1_u4 = Constraint(m.PERIOD,rule = bus02_1_u4)

def bus02_2_R(m,t):
    return m.t02_2[3,t] <= RoCoF
m.bus02_2_R = Constraint(m.PERIOD, rule = bus02_2_R)

def bus02_2_l1(m,t):
    return bus2_seg1[0]*100*m.Pg[6,t]*m.ug[6,t] + bus2_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1[2]* m.inertia_constant[6]*m.genD_Pmax[6]*m.ug[6,t]) + bus2_seg1[3]<= m.t02_2[1,t]
m.bus02_2_l1 = Constraint(m.PERIOD, rule = bus02_2_l1)

def bus02_2_u1(m,t):
    return bus2_seg1[0]*100*m.Pg[6,t]*m.ug[6,t] + bus2_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1[2]* m.inertia_constant[6]*m.genD_Pmax[6]*m.ug[6,t]) + bus2_seg1[3] + m.v02_2[1,t]*A >= m.t02_2[1,t]
m.bus02_2_u1 = Constraint(m.PERIOD, rule = bus02_2_u1)

def bus02_2_l2(m,t):
    return bus2_seg2[0]*100*m.Pg[6,t]*m.ug[6,t] + bus2_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2[2]* m.inertia_constant[6]*m.genD_Pmax[6]*m.ug[6,t]) + bus2_seg2[3] <= m.t02_2[1,t]
m.bus02_2_l2 = Constraint(m.PERIOD,rule = bus02_2_l2)

def bus02_2_u2(m,t):
    return bus2_seg2[0]*100*m.Pg[6,t]*m.ug[6,t] + bus2_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2[2]* m.inertia_constant[6]*m.genD_Pmax[6]*m.ug[6,t]) + bus2_seg2[3]+ (1 - m.v02_2[1,t])*A >= m.t02_2[1,t]
m.bus02_2_u2 = Constraint(m.PERIOD,rule = bus02_2_u2)

def bus02_2_t12(m,t):
    return m.t02_2[1,t] <= m.t02_2[2,t]
m.bus02_2_t12 = Constraint(m.PERIOD,rule = bus02_2_t12)

def bus02_2_t21(m,t):
    return m.t02_2[2,t]<=m.t02_2[1,t] + m.v02_2[2,t]*A
m.bus02_2_t21 = Constraint(m.PERIOD,rule = bus02_2_t21)

def bus02_2_l3(m,t):
    return bus2_seg3[0]*100*m.Pg[6,t]*m.ug[6,t] + bus2_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3[2]* m.inertia_constant[6]*m.genD_Pmax[6]*m.ug[6,t]) + bus2_seg3[3] <= m.t02_2[2,t]
m.bus02_2_l3 = Constraint(m.PERIOD,rule = bus02_2_l3)

def bus02_2_u3(m,t):
    return bus2_seg3[0]*100*m.Pg[6,t]*m.ug[6,t] + bus2_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3[2]* m.inertia_constant[6]*m.genD_Pmax[6]*m.ug[6,t]) + bus2_seg3[3] + (1 - m.v02_2[2,t])*A >= m.t02_2[2,t]
m.bus02_2_u3 = Constraint(m.PERIOD,rule = bus02_2_u3)

def bus02_2_t23(m,t):
    return m.t02_2[2,t] <= m.t02_2[3,t]
m.bus02_2_t23 = Constraint(m.PERIOD,rule = bus02_2_t23)

def bus02_2_t32(m,t):
    return m.t02_2[3,t]<=m.t02_2[2,t] + m.v02_2[3,t]*A
m.bus02_2_t32 = Constraint(m.PERIOD,rule = bus02_2_t32)

def bus02_2_l4(m,t):
    return bus2_seg4[0]*100*m.Pg[6,t]*m.ug[6,t] + bus2_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4[2]* m.inertia_constant[6]*m.genD_Pmax[6]*m.ug[6,t]) + bus2_seg4[3] <= m.t02_2[3,t]
m.bus02_2_l4 = Constraint(m.PERIOD,rule = bus02_2_l4)

def bus02_2_u4(m,t):
    return bus2_seg4[0]*100*m.Pg[6,t]*m.ug[6,t] + bus2_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4[2]* m.inertia_constant[6]*m.genD_Pmax[6]*m.ug[6,t]) + bus2_seg4[3] + (1 - m.v02_2[3,t])*A >= m.t02_2[3,t]
m.bus02_2_u4 = Constraint(m.PERIOD,rule = bus02_2_u4)

def bus02_3_R(m,t):
    return m.t02_3[3,t] <= RoCoF
m.bus02_3_R = Constraint(m.PERIOD, rule = bus02_3_R)

def bus02_3_l1(m,t):
    return bus2_seg1[0]*100*m.Pg[7,t]*m.ug[7,t] + bus2_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1[2]* m.inertia_constant[7]*m.genD_Pmax[7]*m.ug[7,t]) + bus2_seg1[3] <= m.t02_3[1,t]
m.bus02_3_l1 = Constraint(m.PERIOD, rule = bus02_3_l1)

def bus02_3_u1(m,t):
    return bus2_seg1[0]*100*m.Pg[7,t]*m.ug[7,t] + bus2_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1[2]* m.inertia_constant[7]*m.genD_Pmax[7]*m.ug[7,t]) + bus2_seg1[3] + m.v02_3[1,t]*A >= m.t02_3[1,t]
m.bus02_3_u1 = Constraint(m.PERIOD, rule = bus02_3_u1)

def bus02_3_l2(m,t):
    return bus2_seg2[0]*100*m.Pg[7,t]*m.ug[7,t] + bus2_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2[2]* m.inertia_constant[7]*m.genD_Pmax[7]*m.ug[7,t]) + bus2_seg2[3] <= m.t02_3[1,t]
m.bus02_3_l2 = Constraint(m.PERIOD,rule = bus02_3_l2)

def bus02_3_u2(m,t):
    return bus2_seg2[0]*100*m.Pg[7,t]*m.ug[7,t] + bus2_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2[2]* m.inertia_constant[7]*m.genD_Pmax[7]*m.ug[7,t]) + bus2_seg2[3]+ (1 - m.v02_3[1,t])*A >= m.t02_3[1,t]
m.bus02_3_u2 = Constraint(m.PERIOD,rule = bus02_3_u2)

def bus02_3_t12(m,t):
    return m.t02_3[1,t] <= m.t02_3[2,t]
m.bus02_3_t12 = Constraint(m.PERIOD,rule = bus02_3_t12)

def bus02_3_t21(m,t):
    return m.t02_3[2,t]<=m.t02_3[1,t] + m.v02_3[2,t]*A
m.bus02_3_t21 = Constraint(m.PERIOD,rule = bus02_3_t21)

def bus02_3_l3(m,t):
    return bus2_seg3[0]*100*m.Pg[7,t]*m.ug[7,t] + bus2_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3[2]* m.inertia_constant[7]*m.genD_Pmax[7]*m.ug[7,t]) + bus2_seg3[3]  <= m.t02_3[2,t]
m.bus02_3_l3 = Constraint(m.PERIOD,rule = bus02_3_l3)

def bus02_3_u3(m,t):
    return bus2_seg3[0]*100*m.Pg[7,t]*m.ug[7,t] + bus2_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3[2]* m.inertia_constant[7]*m.genD_Pmax[7]*m.ug[7,t]) + bus2_seg3[3] + (1 - m.v02_3[2,t])*A >= m.t02_3[2,t]
m.bus02_3_u3 = Constraint(m.PERIOD,rule = bus02_3_u3)

def bus02_3_t23(m,t):
    return m.t02_3[2,t] <= m.t02_3[3,t]
m.bus02_3_t23 = Constraint(m.PERIOD,rule = bus02_3_t23)

def bus02_3_t32(m,t):
    return m.t02_3[3,t]<=m.t02_3[2,t] + m.v02_3[3,t]*A
m.bus02_3_t32 = Constraint(m.PERIOD,rule = bus02_3_t32)

def bus02_3_l4(m,t):
    return bus2_seg4[0]*100*m.Pg[7,t]*m.ug[7,t] + bus2_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4[2]* m.inertia_constant[7]*m.genD_Pmax[7]*m.ug[7,t]) + bus2_seg4[3] <= m.t02_3[3,t]
m.bus02_3_l4 = Constraint(m.PERIOD,rule = bus02_3_l4)

def bus02_3_u4(m,t):
    return bus2_seg4[0]*100*m.Pg[7,t]*m.ug[7,t] + bus2_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4[2]* m.inertia_constant[7]*m.genD_Pmax[7]*m.ug[7,t]) + bus2_seg4[3] + (1 - m.v02_3[3,t])*A >= m.t02_3[3,t]
m.bus02_3_u4 = Constraint(m.PERIOD,rule = bus02_3_u4)

def bus02_4_R(m,t):
    return m.t02_4[3,t] <= RoCoF
m.bus02_4_R = Constraint(m.PERIOD, rule = bus02_4_R)

def bus02_4_l1(m,t):
    return bus2_seg1[0]*100*m.Pg[8,t]*m.ug[8,t] + bus2_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1[2]* m.inertia_constant[8]*m.genD_Pmax[8]*m.ug[8,t]) + bus2_seg1[3] <= m.t02_4[1,t]
m.bus02_4_l1 = Constraint(m.PERIOD, rule = bus02_4_l1)

def bus02_4_u1(m,t):
    return bus2_seg1[0]*100*m.Pg[8,t]*m.ug[8,t] + bus2_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1[2]* m.inertia_constant[8]*m.genD_Pmax[8]*m.ug[8,t]) + bus2_seg1[3] + m.v02_4[1,t]*A >= m.t02_4[1,t]
m.bus02_4_u1 = Constraint(m.PERIOD, rule = bus02_4_u1)

def bus02_4_l2(m,t):
    return bus2_seg2[0]*100*m.Pg[8,t]*m.ug[8,t] + bus2_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2[2]* m.inertia_constant[8]*m.genD_Pmax[8]*m.ug[8,t]) + bus2_seg2[3] <= m.t02_4[1,t]
m.bus02_4_l2 = Constraint(m.PERIOD,rule = bus02_4_l2)

def bus02_4_u2(m,t):
    return bus2_seg2[0]*100*m.Pg[8,t]*m.ug[8,t] + bus2_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2[2]* m.inertia_constant[8]*m.genD_Pmax[8]*m.ug[8,t]) + bus2_seg2[3] + (1 - m.v02_4[1,t])*A >= m.t02_4[1,t]
m.bus02_4_u2 = Constraint(m.PERIOD,rule = bus02_4_u2)

def bus02_4_t12(m,t):
    return m.t02_4[1,t] <= m.t02_4[2,t]
m.bus02_4_t12 = Constraint(m.PERIOD,rule = bus02_4_t12)

def bus02_4_t21(m,t):
    return m.t02_4[2,t]<=m.t02_4[1,t] + m.v02_4[2,t]*A
m.bus02_4_t21 = Constraint(m.PERIOD,rule = bus02_4_t21)

def bus02_4_l3(m,t):
    return bus2_seg3[0]*100*m.Pg[8,t]*m.ug[8,t] + bus2_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3[2]* m.inertia_constant[8]*m.genD_Pmax[8]*m.ug[8,t]) + bus2_seg3[3]  <= m.t02_4[2,t]
m.bus02_4_l3 = Constraint(m.PERIOD,rule = bus02_4_l3)

def bus02_4_u3(m,t):
    return bus2_seg3[0]*100*m.Pg[8,t]*m.ug[8,t] + bus2_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3[2]* m.inertia_constant[8]*m.genD_Pmax[8]*m.ug[8,t]) + bus2_seg3[3] + (1 - m.v02_4[2,t])*A >= m.t02_4[2,t]
m.bus02_4_u3 = Constraint(m.PERIOD,rule = bus02_4_u3)

def bus02_4_t23(m,t):
    return m.t02_4[2,t] <= m.t02_4[3,t]
m.bus02_4_t23 = Constraint(m.PERIOD,rule = bus02_4_t23)

def bus02_4_t32(m,t):
    return m.t02_4[3,t]<=m.t02_4[2,t] + m.v02_4[3,t]*A
m.bus02_4_t32 = Constraint(m.PERIOD,rule = bus02_4_t32)

def bus02_4_l4(m,t):
    return bus2_seg4[0]*100*m.Pg[8,t]*m.ug[8,t] + bus2_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4[2]* m.inertia_constant[8]*m.genD_Pmax[8]*m.ug[8,t]) + bus2_seg4[3] <= m.t02_4[3,t]
m.bus02_4_l4 = Constraint(m.PERIOD,rule = bus02_4_l4)

def bus02_4_u4(m,t):
    return bus2_seg4[0]*100*m.Pg[8,t]*m.ug[8,t] + bus2_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4[2]* m.inertia_constant[8]*m.genD_Pmax[8]*m.ug[8,t]) + bus2_seg4[3] + (1 - m.v02_4[3,t])*A >= m.t02_4[3,t]
m.bus02_4_u4 = Constraint(m.PERIOD,rule = bus02_4_u4)

def bus02_5_R(m,t):
    return m.t02_5[3,t] <= RoCoF
m.bus02_5_R = Constraint(m.PERIOD, rule = bus02_5_R)

def bus02_5_l1(m,t):
    return bus2_seg1[0]*100*m.Pg[9,t]*m.ug[9,t] + bus2_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1[2]* m.inertia_constant[9]*m.genD_Pmax[9]*m.ug[9,t]) + bus2_seg1[3] <= m.t02_5[1,t]
m.bus02_5_l1 = Constraint(m.PERIOD, rule = bus02_5_l1)

def bus02_5_u1(m,t):
    return bus2_seg1[0]*100*m.Pg[9,t]*m.ug[9,t] + bus2_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1[2]* m.inertia_constant[9]*m.genD_Pmax[9]*m.ug[9,t]) + bus2_seg1[3] + m.v02_5[1,t]*A >= m.t02_5[1,t]
m.bus02_5_u1 = Constraint(m.PERIOD, rule = bus02_5_u1)

def bus02_5_l2(m,t):
    return bus2_seg2[0]*100*m.Pg[9,t]*m.ug[9,t] + bus2_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2[2]* m.inertia_constant[9]*m.genD_Pmax[9]*m.ug[9,t]) + bus2_seg2[3] <= m.t02_5[1,t]
m.bus02_5_l2 = Constraint(m.PERIOD,rule = bus02_5_l2)

def bus02_5_u2(m,t):
    return bus2_seg2[0]*100*m.Pg[9,t]*m.ug[9,t] + bus2_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2[2]* m.inertia_constant[9]*m.genD_Pmax[9]*m.ug[9,t]) + bus2_seg2[3] + (1 - m.v02_5[1,t])*A >= m.t02_5[1,t]
m.bus02_5_u2 = Constraint(m.PERIOD,rule = bus02_5_u2)

def bus02_5_t12(m,t):
    return m.t02_5[1,t] <= m.t02_5[2,t]
m.bus02_5_t12 = Constraint(m.PERIOD,rule = bus02_5_t12)

def bus02_5_t21(m,t):
    return m.t02_5[2,t]<=m.t02_5[1,t] + m.v02_5[2,t]*A
m.bus02_5_t21 = Constraint(m.PERIOD,rule = bus02_5_t21)

def bus02_5_l3(m,t):
    return bus2_seg3[0]*100*m.Pg[9,t]*m.ug[9,t] + bus2_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3[2]* m.inertia_constant[9]*m.genD_Pmax[9]*m.ug[9,t]) + bus2_seg3[3]  <= m.t02_5[2,t]
m.bus02_5_l3 = Constraint(m.PERIOD,rule = bus02_5_l3)

def bus02_5_u3(m,t):
    return bus2_seg3[0]*100*m.Pg[9,t]*m.ug[9,t] + bus2_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3[2]* m.inertia_constant[9]*m.genD_Pmax[9]*m.ug[9,t]) + bus2_seg3[3] + (1 - m.v02_5[2,t])*A >= m.t02_5[2,t]
m.bus02_5_u3 = Constraint(m.PERIOD,rule = bus02_5_u3)

def bus02_5_t23(m,t):
    return m.t02_5[2,t] <= m.t02_5[3,t]
m.bus02_5_t23 = Constraint(m.PERIOD,rule = bus02_5_t23)

def bus02_5_t32(m,t):
    return m.t02_5[3,t]<=m.t02_5[2,t] + m.v02_5[3,t]*A
m.bus02_5_t32 = Constraint(m.PERIOD,rule = bus02_5_t32)

def bus02_5_l4(m,t):
    return bus2_seg4[0]*100*m.Pg[9,t]*m.ug[9,t] + bus2_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4[2]* m.inertia_constant[9]*m.genD_Pmax[9]*m.ug[9,t]) + bus2_seg4[3] <= m.t02_5[3,t]
m.bus02_5_l4 = Constraint(m.PERIOD,rule = bus02_5_l4)

def bus02_5_u4(m,t):
    return bus2_seg4[0]*100*m.Pg[9,t]*m.ug[9,t] + bus2_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4[2]* m.inertia_constant[9]*m.genD_Pmax[9]*m.ug[9,t]) + bus2_seg4[3] + (1 - m.v02_5[3,t])*A >= m.t02_5[3,t]
m.bus02_5_u4 = Constraint(m.PERIOD,rule = bus02_5_u4)

def bus02_6_R(m,t):
    return m.t02_6[3,t] <= RoCoF
m.bus02_6_R = Constraint(m.PERIOD, rule = bus02_6_R)

def bus02_6_l1(m,t):
    return bus2_seg1[0]*100*m.Pg[10,t]*m.ug[10,t] + bus2_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1[2]* m.inertia_constant[10]*m.genD_Pmax[10]*m.ug[10,t]) + bus2_seg1[3] <= m.t02_6[1,t]
m.bus02_6_l1 = Constraint(m.PERIOD, rule = bus02_6_l1)

def bus02_6_u1(m,t):
    return bus2_seg1[0]*100*m.Pg[10,t]*m.ug[10,t] + bus2_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1[2]* m.inertia_constant[10]*m.genD_Pmax[10]*m.ug[10,t]) + bus2_seg1[3] + m.v02_6[1,t]*A >= m.t02_6[1,t]
m.bus02_6_u1 = Constraint(m.PERIOD, rule = bus02_6_u1)

def bus02_6_l2(m,t):
    return bus2_seg2[0]*100*m.Pg[10,t]*m.ug[10,t] + bus2_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2[2]* m.inertia_constant[10]*m.genD_Pmax[10]*m.ug[10,t]) + bus2_seg2[3] <= m.t02_6[1,t]
m.bus02_6_l2 = Constraint(m.PERIOD,rule = bus02_6_l2)

def bus02_6_u2(m,t):
    return bus2_seg2[0]*100*m.Pg[10,t]*m.ug[10,t] + bus2_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2[2]* m.inertia_constant[10]*m.genD_Pmax[10]*m.ug[10,t]) + bus2_seg2[3] + (1 - m.v02_6[1,t])*A >= m.t02_6[1,t]
m.bus02_6_u2 = Constraint(m.PERIOD,rule = bus02_6_u2)

def bus02_6_t12(m,t):
    return m.t02_6[1,t] <= m.t02_6[2,t]
m.bus02_6_t12 = Constraint(m.PERIOD,rule = bus02_6_t12)

def bus02_6_t21(m,t):
    return m.t02_6[2,t]<=m.t02_6[1,t] + m.v02_6[2,t]*A
m.bus02_6_t21 = Constraint(m.PERIOD,rule = bus02_6_t21)

def bus02_6_l3(m,t):
    return bus2_seg3[0]*100*m.Pg[10,t]*m.ug[10,t] + bus2_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3[2]* m.inertia_constant[10]*m.genD_Pmax[10]*m.ug[10,t]) + bus2_seg3[3]  <= m.t02_6[2,t]
m.bus02_6_l3 = Constraint(m.PERIOD,rule = bus02_6_l3)

def bus02_6_u3(m,t):
    return bus2_seg3[0]*100*m.Pg[10,t]*m.ug[10,t] + bus2_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3[2]* m.inertia_constant[10]*m.genD_Pmax[10]*m.ug[10,t]) + bus2_seg3[3] + (1 - m.v02_6[2,t])*A >= m.t02_6[2,t]
m.bus02_6_u3 = Constraint(m.PERIOD,rule = bus02_6_u3)

def bus02_6_t23(m,t):
    return m.t02_6[2,t] <= m.t02_6[3,t]
m.bus02_6_t23 = Constraint(m.PERIOD,rule = bus02_6_t23)

def bus02_6_t32(m,t):
    return m.t02_6[3,t]<=m.t02_6[2,t] + m.v02_6[3,t]*A
m.bus02_6_t32 = Constraint(m.PERIOD,rule = bus02_6_t32)

def bus02_6_l4(m,t):
    return bus2_seg4[0]*100*m.Pg[10,t]*m.ug[10,t] + bus2_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4[2]* m.inertia_constant[10]*m.genD_Pmax[10]*m.ug[10,t]) + bus2_seg4[3] <= m.t02_6[3,t]
m.bus02_6_l4 = Constraint(m.PERIOD,rule = bus02_6_l4)

def bus02_6_u4(m,t):
    return bus2_seg4[0]*100*m.Pg[10,t]*m.ug[10,t] + bus2_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4[2]* m.inertia_constant[10]*m.genD_Pmax[10]*m.ug[10,t]) + bus2_seg4[3] + (1 - m.v02_6[3,t])*A >= m.t02_6[3,t]
m.bus02_6_u4 = Constraint(m.PERIOD,rule = bus02_6_u4)

# Node 07 locational RoCoF constraints

def bus07_1_R(m,t):
    return m.t07_1[3,t] <= RoCoF
m.bus07_1_R = Constraint(m.PERIOD, rule = bus07_1_R)

def bus07_1_l1(m,t):
    return bus7_seg1[0]*100*m.Pg[11,t]*m.ug[11,t] + bus7_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg1[2]* m.inertia_constant[11]*m.genD_Pmax[11]*m.ug[11,t]) + bus7_seg1[3]<= m.t07_1[1,t]
m.bus07_1_l1 = Constraint(m.PERIOD, rule = bus07_1_l1)

def bus07_1_u1(m,t):
    return bus7_seg1[0]*100*m.Pg[11,t]*m.ug[11,t] + bus7_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg1[2]* m.inertia_constant[11]*m.genD_Pmax[11]*m.ug[11,t]) + bus7_seg1[3] + m.v07_1[1,t]*A >= m.t07_1[1,t]
m.bus07_1_u1 = Constraint(m.PERIOD, rule = bus07_1_u1)

def bus07_1_l2(m,t):
    return bus7_seg2[0]*100*m.Pg[11,t]*m.ug[11,t] + bus7_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg2[2]* m.inertia_constant[11]*m.genD_Pmax[11]*m.ug[11,t]) + bus7_seg2[3] <= m.t07_1[1,t]
m.bus07_1_l2 = Constraint(m.PERIOD,rule = bus07_1_l2)

def bus07_1_u2(m,t):
    return bus7_seg2[0]*100*m.Pg[11,t]*m.ug[11,t] + bus7_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg2[2]* m.inertia_constant[11]*m.genD_Pmax[11]*m.ug[11,t]) + bus7_seg2[3] + (1 - m.v07_1[1,t])*A >= m.t07_1[1,t]
m.bus07_1_u2 = Constraint(m.PERIOD,rule = bus07_1_u2)

def bus07_1_t12(m,t):
    return m.t07_1[1,t] <= m.t07_1[2,t]
m.bus07_1_t12 = Constraint(m.PERIOD,rule = bus07_1_t12)

def bus07_1_t21(m,t):
    return m.t07_1[2,t]<=m.t07_1[1,t] + m.v07_1[2,t]*A
m.bus07_1_t21 = Constraint(m.PERIOD,rule = bus07_1_t21)

def bus07_1_l3(m,t):
    return bus7_seg3[0]*100*m.Pg[11,t]*m.ug[11,t] + bus7_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg3[2]* m.inertia_constant[11]*m.genD_Pmax[11]*m.ug[11,t]) + bus7_seg3[3] <= m.t07_1[2,t]
m.bus07_1_l3 = Constraint(m.PERIOD,rule = bus07_1_l3)

def bus07_1_u3(m,t):
    return bus7_seg3[0]*100*m.Pg[11,t]*m.ug[11,t] + bus7_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg3[2]* m.inertia_constant[11]*m.genD_Pmax[11]*m.ug[11,t]) + bus7_seg3[3]+ (1 - m.v07_1[2,t])*A >= m.t07_1[2,t]
m.bus07_1_u3 = Constraint(m.PERIOD,rule = bus07_1_u3)

def bus07_1_t23(m,t):
    return m.t07_1[2,t] <= m.t07_1[3,t]
m.bus07_1_t23 = Constraint(m.PERIOD,rule = bus07_1_t23)

def bus07_1_t32(m,t):
    return m.t07_1[3,t]<=m.t07_1[2,t] + m.v07_1[3,t]*A
m.bus07_1_t32 = Constraint(m.PERIOD,rule = bus07_1_t32)

def bus07_1_l4(m,t):
    return bus7_seg4[0]*100*m.Pg[11,t]*m.ug[11,t] + bus7_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg4[2]* m.inertia_constant[11]*m.genD_Pmax[11]*m.ug[11,t]) + bus7_seg4[3] <= m.t07_1[3,t]
m.bus07_1_l4 = Constraint(m.PERIOD,rule = bus07_1_l4)

def bus07_1_u4(m,t):
    return bus7_seg4[0]*100*m.Pg[11,t]*m.ug[11,t] + bus7_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg4[2]* m.inertia_constant[11]*m.genD_Pmax[11]*m.ug[11,t]) + bus7_seg4[3] + (1 - m.v07_1[3,t])*A >= m.t07_1[3,t]
m.bus07_1_u4 = Constraint(m.PERIOD,rule = bus07_1_u4)

def bus07_2_R(m,t):
    return m.t07_2[3,t] <= RoCoF
m.bus07_2_R = Constraint(m.PERIOD, rule = bus07_2_R)

def bus07_2_l1(m,t):
    return bus7_seg1[0]*100*m.Pg[12,t]*m.ug[12,t] + bus7_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg1[2]* m.inertia_constant[12]*m.genD_Pmax[12]*m.ug[12,t]) + bus7_seg1[3] <= m.t07_2[1,t]
m.bus07_2_l1 = Constraint(m.PERIOD, rule = bus07_2_l1)

def bus07_2_u1(m,t):
    return bus7_seg1[0]*100*m.Pg[12,t]*m.ug[12,t] + bus7_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg1[2]* m.inertia_constant[12]*m.genD_Pmax[12]*m.ug[12,t]) + bus7_seg1[3]+ m.v07_2[1,t]*A >= m.t07_2[1,t]
m.bus07_2_u1 = Constraint(m.PERIOD, rule = bus07_2_u1)

def bus07_2_l2(m,t):
    return bus7_seg2[0]*100*m.Pg[12,t]*m.ug[12,t] + bus7_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg2[2]* m.inertia_constant[12]*m.genD_Pmax[12]*m.ug[12,t]) + bus7_seg2[3] <= m.t07_2[1,t]
m.bus07_2_l2 = Constraint(m.PERIOD,rule = bus07_2_l2)

def bus07_2_u2(m,t):
    return bus7_seg2[0]*100*m.Pg[12,t]*m.ug[12,t] + bus7_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg2[2]* m.inertia_constant[12]*m.genD_Pmax[12]*m.ug[12,t]) + bus7_seg2[3] + (1 - m.v07_2[1,t])*A >= m.t07_2[1,t]
m.bus07_2_u2 = Constraint(m.PERIOD,rule = bus07_2_u2)

def bus07_2_t12(m,t):
    return m.t07_2[1,t] <= m.t07_2[2,t]
m.bus07_2_t12 = Constraint(m.PERIOD,rule = bus07_2_t12)

def bus07_2_t21(m,t):
    return m.t07_2[2,t]<=m.t07_2[1,t] + m.v07_2[2,t]*A
m.bus07_2_t21 = Constraint(m.PERIOD,rule = bus07_2_t21)

def bus07_2_l3(m,t):
    return bus7_seg3[0]*100*m.Pg[12,t]*m.ug[12,t] + bus7_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg3[2]* m.inertia_constant[12]*m.genD_Pmax[12]*m.ug[12,t]) + bus7_seg3[3] <= m.t07_2[2,t]
m.bus07_2_l3 = Constraint(m.PERIOD,rule = bus07_2_l3)

def bus07_2_u3(m,t):
    return bus7_seg3[0]*100*m.Pg[12,t]*m.ug[12,t] + bus7_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg3[2]* m.inertia_constant[12]*m.genD_Pmax[12]*m.ug[12,t]) + bus7_seg3[3] + (1 - m.v07_2[2,t])*A >= m.t07_2[2,t]
m.bus07_2_u3 = Constraint(m.PERIOD,rule = bus07_2_u3)

def bus07_2_t23(m,t):
    return m.t07_2[2,t] <= m.t07_2[3,t]
m.bus07_2_t23 = Constraint(m.PERIOD,rule = bus07_2_t23)

def bus07_2_t32(m,t):
    return m.t07_2[3,t]<=m.t07_2[2,t] + m.v07_2[3,t]*A
m.bus07_2_t32 = Constraint(m.PERIOD,rule = bus07_2_t32)

def bus07_2_l4(m,t):
    return bus7_seg4[0]*100*m.Pg[12,t]*m.ug[12,t] + bus7_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg4[2]* m.inertia_constant[12]*m.genD_Pmax[12]*m.ug[12,t]) + bus7_seg4[3] <= m.t07_2[3,t]
m.bus07_2_l4 = Constraint(m.PERIOD,rule = bus07_2_l4)

def bus07_2_u4(m,t):
    return bus7_seg4[0]*100*m.Pg[12,t]*m.ug[12,t] + bus7_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg4[2]* m.inertia_constant[12]*m.genD_Pmax[12]*m.ug[12,t]) + bus7_seg4[3] + (1 - m.v07_2[3,t])*A >= m.t07_2[3,t]
m.bus07_2_u4 = Constraint(m.PERIOD,rule = bus07_2_u4)

def bus07_3_R(m,t):
    return m.t07_3[3,t] <= RoCoF
m.bus07_3_R = Constraint(m.PERIOD, rule = bus07_3_R)

def bus07_3_l1(m,t):
    return bus7_seg1[0]*100*m.Pg[13,t]*m.ug[13,t] + bus7_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg1[2]* m.inertia_constant[13]*m.genD_Pmax[13]*m.ug[13,t]) + bus7_seg1[3] <= m.t07_3[1,t]
m.bus07_3_l1 = Constraint(m.PERIOD, rule = bus07_3_l1)

def bus07_3_u1(m,t):
    return bus7_seg1[0]*100*m.Pg[13,t]*m.ug[13,t] + bus7_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg1[2]* m.inertia_constant[13]*m.genD_Pmax[13]*m.ug[13,t]) + bus7_seg1[3] + m.v07_3[1,t]*A >= m.t07_3[1,t]
m.bus07_3_u1 = Constraint(m.PERIOD, rule = bus07_3_u1)

def bus07_3_l2(m,t):
    return bus7_seg2[0]*100*m.Pg[13,t]*m.ug[13,t] + bus7_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg2[2]* m.inertia_constant[13]*m.genD_Pmax[13]*m.ug[13,t]) + bus7_seg2[3] <= m.t07_3[1,t]
m.bus07_3_l2 = Constraint(m.PERIOD,rule = bus07_3_l2)

def bus07_3_u2(m,t):
    return bus7_seg2[0]*100*m.Pg[13,t]*m.ug[13,t] + bus7_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg2[2]* m.inertia_constant[13]*m.genD_Pmax[13]*m.ug[13,t]) + bus7_seg2[3] + (1 - m.v07_3[1,t])*A >= m.t07_3[1,t]
m.bus07_3_u2 = Constraint(m.PERIOD,rule = bus07_3_u2)

def bus07_3_t12(m,t):
    return m.t07_3[1,t] <= m.t07_3[2,t]
m.bus07_3_t12 = Constraint(m.PERIOD,rule = bus07_3_t12)

def bus07_3_t21(m,t):
    return m.t07_3[2,t]<=m.t07_3[1,t] + m.v07_3[2,t]*A
m.bus07_3_t21 = Constraint(m.PERIOD,rule = bus07_3_t21)

def bus07_3_l3(m,t):
    return bus7_seg3[0]*100*m.Pg[13,t]*m.ug[13,t] + bus7_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg3[2]* m.inertia_constant[13]*m.genD_Pmax[13]*m.ug[13,t]) + bus7_seg3[3]  <= m.t07_3[2,t]
m.bus07_3_l3 = Constraint(m.PERIOD,rule = bus07_3_l3)

def bus07_3_u3(m,t):
    return bus7_seg3[0]*100*m.Pg[13,t]*m.ug[13,t] + bus7_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg3[2]* m.inertia_constant[13]*m.genD_Pmax[13]*m.ug[13,t]) + bus7_seg3[3]  + (1 - m.v07_3[2,t])*A >= m.t07_3[2,t]
m.bus07_3_u3 = Constraint(m.PERIOD,rule = bus07_3_u3)

def bus07_3_t23(m,t):
    return m.t07_3[2,t] <= m.t07_3[3,t]
m.bus07_3_t23 = Constraint(m.PERIOD,rule = bus07_3_t23)

def bus07_3_t32(m,t):
    return m.t07_3[3,t]<=m.t07_3[2,t] + m.v07_3[3,t]*A
m.bus07_3_t32 = Constraint(m.PERIOD,rule = bus07_3_t32)

def bus07_3_l4(m,t):
    return bus7_seg4[0]*100*m.Pg[13,t]*m.ug[13,t] + bus7_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg4[2]* m.inertia_constant[13]*m.genD_Pmax[13]*m.ug[13,t]) + bus7_seg4[3] <= m.t07_3[3,t]
m.bus07_3_l4 = Constraint(m.PERIOD,rule = bus07_3_l4)

def bus07_3_u4(m,t):
    return bus7_seg4[0]*100*m.Pg[13,t]*m.ug[13,t] + bus7_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg4[2]* m.inertia_constant[13]*m.genD_Pmax[13]*m.ug[13,t]) + bus7_seg4[3] + (1 - m.v07_3[3,t])*A >= m.t07_3[3,t]
m.bus07_3_u4 = Constraint(m.PERIOD,rule = bus07_3_u4)

# Node 13 locational RoCoF constraints

def bus13_1_R(m,t):
    return m.t13_1[3,t] <= RoCoF
m.bus13_1_R = Constraint(m.PERIOD, rule = bus13_1_R)

def bus13_1_l1(m,t):
    return bus13_seg1[0]*100*m.Pg[14,t]*m.ug[14,t] + bus13_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1[2]* m.inertia_constant[14]*m.genD_Pmax[14]*m.ug[14,t]) + bus13_seg1[3] <= m.t13_1[1,t]
m.bus13_1_l1 = Constraint(m.PERIOD, rule = bus13_1_l1)

def bus13_1_u1(m,t):
    return bus13_seg1[0]*100*m.Pg[14,t]*m.ug[14,t] + bus13_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1[2]* m.inertia_constant[14]*m.genD_Pmax[14]*m.ug[14,t]) + bus13_seg1[3] + m.v13_1[1,t]*A >= m.t13_1[1,t]
m.bus13_1_u1 = Constraint(m.PERIOD, rule = bus13_1_u1)

def bus13_1_l2(m,t):
    return bus13_seg2[0]*100*m.Pg[14,t]*m.ug[14,t] + bus13_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2[2]* m.inertia_constant[14]*m.genD_Pmax[14]*m.ug[14,t]) + bus13_seg2[3] <= m.t13_1[1,t]
m.bus13_1_l2 = Constraint(m.PERIOD,rule = bus13_1_l2)

def bus13_1_u2(m,t):
    return bus13_seg2[0]*100*m.Pg[14,t]*m.ug[14,t] + bus13_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2[2]* m.inertia_constant[14]*m.genD_Pmax[14]*m.ug[14,t]) + bus13_seg2[3] + (1 - m.v13_1[1,t])*A >= m.t13_1[1,t]
m.bus13_1_u2 = Constraint(m.PERIOD,rule = bus13_1_u2)

def bus13_1_t12(m,t):
    return m.t13_1[1,t] <= m.t13_1[2,t]
m.bus13_1_t12 = Constraint(m.PERIOD,rule = bus13_1_t12)

def bus13_1_t21(m,t):
    return m.t13_1[2,t]<=m.t13_1[1,t] + m.v13_1[2,t]*A
m.bus13_1_t21 = Constraint(m.PERIOD,rule = bus13_1_t21)

def bus13_1_l3(m,t):
    return bus13_seg3[0]*100*m.Pg[14,t]*m.ug[14,t] + bus13_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3[2]* m.inertia_constant[14]*m.genD_Pmax[14]*m.ug[14,t]) + bus13_seg3[3] <= m.t13_1[2,t]
m.bus13_1_l3 = Constraint(m.PERIOD,rule = bus13_1_l3)

def bus13_1_u3(m,t):
    return bus13_seg3[0]*100*m.Pg[14,t]*m.ug[14,t] + bus13_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3[2]* m.inertia_constant[14]*m.genD_Pmax[14]*m.ug[14,t]) + bus13_seg3[3] + (1 - m.v13_1[2,t])*A >= m.t13_1[2,t]
m.bus13_1_u3 = Constraint(m.PERIOD,rule = bus13_1_u3)

def bus13_1_t23(m,t):
    return m.t13_1[2,t] <= m.t13_1[3,t]
m.bus13_1_t23 = Constraint(m.PERIOD,rule = bus13_1_t23)

def bus13_1_t32(m,t):
    return m.t13_1[3,t]<=m.t13_1[2,t] + m.v13_1[3,t]*A
m.bus13_1_t32 = Constraint(m.PERIOD,rule = bus13_1_t32)

def bus13_1_l4(m,t):
    return bus13_seg4[0]*100*m.Pg[14,t]*m.ug[14,t] + bus13_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4[2]* m.inertia_constant[14]*m.genD_Pmax[14]*m.ug[14,t]) + bus13_seg4[3] <= m.t13_1[3,t]
m.bus13_1_l4 = Constraint(m.PERIOD,rule = bus13_1_l4)

def bus13_1_u4(m,t):
    return bus13_seg4[0]*100*m.Pg[14,t]*m.ug[14,t] + bus13_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4[2]* m.inertia_constant[14]*m.genD_Pmax[14]*m.ug[14,t]) + bus13_seg4[3] + (1 - m.v13_1[3,t])*A >= m.t13_1[3,t]
m.bus13_1_u4 = Constraint(m.PERIOD,rule = bus13_1_u4)

def bus13_2_R(m,t):
    return m.t13_2[3,t] <= RoCoF
m.bus13_2_R = Constraint(m.PERIOD, rule = bus13_2_R)

def bus13_2_l1(m,t):
    return bus13_seg1[0]*100*m.Pg[15,t]*m.ug[15,t] + bus13_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1[2]* m.inertia_constant[15]*m.genD_Pmax[15]*m.ug[15,t]) + bus13_seg1[3] <= m.t13_2[1,t]
m.bus13_2_l1 = Constraint(m.PERIOD, rule = bus13_2_l1)

def bus13_2_u1(m,t):
    return bus13_seg1[0]*100*m.Pg[15,t]*m.ug[15,t] + bus13_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1[2]* m.inertia_constant[15]*m.genD_Pmax[15]*m.ug[15,t]) + bus13_seg1[3] + m.v13_2[1,t]*A >= m.t13_2[1,t]
m.bus13_2_u1 = Constraint(m.PERIOD, rule = bus13_2_u1)

def bus13_2_l2(m,t):
    return bus13_seg2[0]*100*m.Pg[15,t]*m.ug[15,t] + bus13_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2[2]* m.inertia_constant[15]*m.genD_Pmax[15]*m.ug[15,t]) + bus13_seg2[3] <= m.t13_2[1,t]
m.bus13_2_l2 = Constraint(m.PERIOD,rule = bus13_2_l2)

def bus13_2_u2(m,t):
    return bus13_seg2[0]*100*m.Pg[15,t]*m.ug[15,t] + bus13_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2[2]* m.inertia_constant[15]*m.genD_Pmax[15]*m.ug[15,t]) + bus13_seg2[3]+ (1 - m.v13_2[1,t])*A >= m.t13_2[1,t]
m.bus13_2_u2 = Constraint(m.PERIOD,rule = bus13_2_u2)

def bus13_2_t12(m,t):
    return m.t13_2[1,t] <= m.t13_2[2,t]
m.bus13_2_t12 = Constraint(m.PERIOD,rule = bus13_2_t12)

def bus13_2_t21(m,t):
    return m.t13_2[2,t]<=m.t13_2[1,t] + m.v13_2[2,t]*A
m.bus13_2_t21 = Constraint(m.PERIOD,rule = bus13_2_t21)

def bus13_2_l3(m,t):
    return bus13_seg3[0]*100*m.Pg[15,t]*m.ug[15,t] + bus13_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3[2]* m.inertia_constant[15]*m.genD_Pmax[15]*m.ug[15,t]) + bus13_seg3[3]  <= m.t13_2[2,t]
m.bus13_2_l3 = Constraint(m.PERIOD,rule = bus13_2_l3)

def bus13_2_u3(m,t):
    return bus13_seg3[0]*100*m.Pg[15,t]*m.ug[15,t] + bus13_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3[2]* m.inertia_constant[15]*m.genD_Pmax[15]*m.ug[15,t]) + bus13_seg3[3] + (1 - m.v13_2[2,t])*A >= m.t13_2[2,t]
m.bus13_2_u3 = Constraint(m.PERIOD,rule = bus13_2_u3)

def bus13_2_t23(m,t):
    return m.t13_2[2,t] <= m.t13_2[3,t]
m.bus13_2_t23 = Constraint(m.PERIOD,rule = bus13_2_t23)

def bus13_2_t32(m,t):
    return m.t13_2[3,t]<=m.t13_2[2,t] + m.v13_2[3,t]*A
m.bus13_2_t32 = Constraint(m.PERIOD,rule = bus13_2_t32)

def bus13_2_l4(m,t):
    return bus13_seg4[0]*100*m.Pg[15,t]*m.ug[15,t] + bus13_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4[2]* m.inertia_constant[15]*m.genD_Pmax[15]*m.ug[15,t]) + bus13_seg4[3] <= m.t13_2[3,t]
m.bus13_2_l4 = Constraint(m.PERIOD,rule = bus13_2_l4)

def bus13_2_u4(m,t):
    return bus13_seg4[0]*100*m.Pg[15,t]*m.ug[15,t] + bus13_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4[2]* m.inertia_constant[15]*m.genD_Pmax[15]*m.ug[15,t]) + bus13_seg4[3] + (1 - m.v13_2[3,t])*A >= m.t13_2[3,t]
m.bus13_2_u4 = Constraint(m.PERIOD,rule = bus13_2_u4)

def bus13_3_R(m,t):
    return m.t13_3[3,t] <= RoCoF
m.bus13_3_R = Constraint(m.PERIOD, rule = bus13_3_R)

def bus13_3_l1(m,t):
    return bus13_seg1[0]*100*m.Pg[16,t]*m.ug[16,t] + bus13_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1[2]* m.inertia_constant[16]*m.genD_Pmax[16]*m.ug[16,t]) + bus13_seg1[3] <= m.t13_3[1,t]
m.bus13_3_l1 = Constraint(m.PERIOD, rule = bus13_3_l1)

def bus13_3_u1(m,t):
    return bus13_seg1[0]*100*m.Pg[16,t]*m.ug[16,t] + bus13_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1[2]* m.inertia_constant[16]*m.genD_Pmax[16]*m.ug[16,t]) + bus13_seg1[3] + m.v13_3[1,t]*A >= m.t13_3[1,t]
m.bus13_3_u1 = Constraint(m.PERIOD, rule = bus13_3_u1)

def bus13_3_l2(m,t):
    return bus13_seg2[0]*100*m.Pg[16,t]*m.ug[16,t] + bus13_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2[2]* m.inertia_constant[16]*m.genD_Pmax[16]*m.ug[16,t]) + bus13_seg2[3] <= m.t13_3[1,t]
m.bus13_3_l2 = Constraint(m.PERIOD,rule = bus13_3_l2)

def bus13_3_u2(m,t):
    return bus13_seg2[0]*100*m.Pg[16,t]*m.ug[16,t] + bus13_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2[2]* m.inertia_constant[16]*m.genD_Pmax[16]*m.ug[16,t]) + bus13_seg2[3] + (1 - m.v13_3[1,t])*A >= m.t13_3[1,t]
m.bus13_3_u2 = Constraint(m.PERIOD,rule = bus13_3_u2)

def bus13_3_t12(m,t):
    return m.t13_3[1,t] <= m.t13_3[2,t]
m.bus13_3_t12 = Constraint(m.PERIOD,rule = bus13_3_t12)

def bus13_3_t21(m,t):
    return m.t13_3[2,t]<=m.t13_3[1,t] + m.v13_3[2,t]*A
m.bus13_3_t21 = Constraint(m.PERIOD,rule = bus13_3_t21)

def bus13_3_l3(m,t):
    return bus13_seg3[0]*100*m.Pg[16,t]*m.ug[16,t] + bus13_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3[2]* m.inertia_constant[16]*m.genD_Pmax[16]*m.ug[16,t]) + bus13_seg3[3] <= m.t13_3[2,t]
m.bus13_3_l3 = Constraint(m.PERIOD,rule = bus13_3_l3)

def bus13_3_u3(m,t):
    return bus13_seg3[0]*100*m.Pg[16,t]*m.ug[16,t] + bus13_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3[2]* m.inertia_constant[16]*m.genD_Pmax[16]*m.ug[16,t]) + bus13_seg3[3] + (1 - m.v13_3[2,t])*A >= m.t13_3[2,t]
m.bus13_3_u3 = Constraint(m.PERIOD,rule = bus13_3_u3)

def bus13_3_t23(m,t):
    return m.t13_3[2,t] <= m.t13_3[3,t]
m.bus13_3_t23 = Constraint(m.PERIOD,rule = bus13_3_t23)

def bus13_3_t32(m,t):
    return m.t13_3[3,t]<=m.t13_3[2,t] + m.v13_3[3,t]*A
m.bus13_3_t32 = Constraint(m.PERIOD,rule = bus13_3_t32)

def bus13_3_l4(m,t):
    return bus13_seg4[0]*100*m.Pg[16,t]*m.ug[16,t] + bus13_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4[2]* m.inertia_constant[16]*m.genD_Pmax[16]*m.ug[16,t]) + bus13_seg4[3] <= m.t13_3[3,t]
m.bus13_3_l4 = Constraint(m.PERIOD,rule = bus13_3_l4)

def bus13_3_u4(m,t):
    return bus13_seg4[0]*100*m.Pg[16,t]*m.ug[16,t] + bus13_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4[2]* m.inertia_constant[16]*m.genD_Pmax[16]*m.ug[16,t]) + bus13_seg4[3] + (1 - m.v13_3[3,t])*A >= m.t13_3[3,t]
m.bus13_3_u4 = Constraint(m.PERIOD,rule = bus13_3_u4)

def bus13_4_R(m,t):
    return m.t13_4[3,t] <= RoCoF
m.bus13_4_R = Constraint(m.PERIOD, rule = bus13_4_R)

def bus13_4_l1(m,t):
    return bus13_seg1[0]*100*m.Pg[17,t]*m.ug[17,t] + bus13_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1[2]* m.inertia_constant[17]*m.genD_Pmax[17]*m.ug[17,t]) + bus13_seg1[3] <= m.t13_4[1,t]
m.bus13_4_l1 = Constraint(m.PERIOD, rule = bus13_4_l1)

def bus13_4_u1(m,t):
    return bus13_seg1[0]*100*m.Pg[17,t]*m.ug[17,t] + bus13_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1[2]* m.inertia_constant[17]*m.genD_Pmax[17]*m.ug[17,t]) + bus13_seg1[3] + m.v13_4[1,t]*A >= m.t13_4[1,t]
m.bus13_4_u1 = Constraint(m.PERIOD, rule = bus13_4_u1)

def bus13_4_l2(m,t):
    return bus13_seg2[0]*100*m.Pg[17,t]*m.ug[17,t] + bus13_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2[2]* m.inertia_constant[17]*m.genD_Pmax[17]*m.ug[17,t]) + bus13_seg2[3] <= m.t13_4[1,t]
m.bus13_4_l2 = Constraint(m.PERIOD,rule = bus13_4_l2)

def bus13_4_u2(m,t):
    return bus13_seg2[0]*100*m.Pg[17,t]*m.ug[17,t] + bus13_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2[2]* m.inertia_constant[17]*m.genD_Pmax[17]*m.ug[17,t]) + bus13_seg2[3]+ (1 - m.v13_4[1,t])*A >= m.t13_4[1,t]
m.bus13_4_u2 = Constraint(m.PERIOD,rule = bus13_4_u2)

def bus13_4_t12(m,t):
    return m.t13_4[1,t] <= m.t13_4[2,t]
m.bus13_4_t12 = Constraint(m.PERIOD,rule = bus13_4_t12)

def bus13_4_t21(m,t):
    return m.t13_4[2,t]<=m.t13_4[1,t] + m.v13_4[2,t]*A
m.bus13_4_t21 = Constraint(m.PERIOD,rule = bus13_4_t21)

def bus13_4_l3(m,t):
    return bus13_seg3[0]*100*m.Pg[17,t]*m.ug[17,t] + bus13_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3[2]* m.inertia_constant[17]*m.genD_Pmax[17]*m.ug[17,t]) + bus13_seg3[3]  <= m.t13_4[2,t]
m.bus13_4_l3 = Constraint(m.PERIOD,rule = bus13_4_l3)

def bus13_4_u3(m,t):
    return bus13_seg3[0]*100*m.Pg[17,t]*m.ug[17,t] + bus13_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3[2]* m.inertia_constant[17]*m.genD_Pmax[17]*m.ug[17,t]) + bus13_seg3[3] + (1 - m.v13_4[2,t])*A >= m.t13_4[2,t]
m.bus13_4_u3 = Constraint(m.PERIOD,rule = bus13_4_u3)

def bus13_4_t23(m,t):
    return m.t13_4[2,t] <= m.t13_4[3,t]
m.bus13_4_t23 = Constraint(m.PERIOD,rule = bus13_4_t23)

def bus13_4_t32(m,t):
    return m.t13_4[3,t]<=m.t13_4[2,t] + m.v13_4[3,t]*A
m.bus13_4_t32 = Constraint(m.PERIOD,rule = bus13_4_t32)

def bus13_4_l4(m,t):
    return bus13_seg4[0]*100*m.Pg[17,t]*m.ug[17,t] + bus13_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4[2]* m.inertia_constant[17]*m.genD_Pmax[17]*m.ug[17,t]) + bus13_seg4[3] <= m.t13_4[3,t]
m.bus13_4_l4 = Constraint(m.PERIOD,rule = bus13_4_l4)

def bus13_4_u4(m,t):
    return bus13_seg4[0]*100*m.Pg[17,t]*m.ug[17,t] + bus13_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4[2]* m.inertia_constant[17]*m.genD_Pmax[17]*m.ug[17,t]) + bus13_seg4[3] + (1 - m.v13_4[3,t])*A >= m.t13_4[3,t]
m.bus13_4_u4 = Constraint(m.PERIOD,rule = bus13_4_u4)

def bus13_5_R(m,t):
    return m.t13_5[3,t] <= RoCoF
m.bus13_5_R = Constraint(m.PERIOD, rule = bus13_5_R)

def bus13_5_l1(m,t):
    return bus13_seg1[0]*100*m.Pg[18,t]*m.ug[18,t] + bus13_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1[2]* m.inertia_constant[18]*m.genD_Pmax[18]*m.ug[18,t]) + bus13_seg1[3] <= m.t13_5[1,t]
m.bus13_5_l1 = Constraint(m.PERIOD, rule = bus13_5_l1)

def bus13_5_u1(m,t):
    return bus13_seg1[0]*100*m.Pg[18,t]*m.ug[18,t] + bus13_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1[2]* m.inertia_constant[18]*m.genD_Pmax[18]*m.ug[18,t]) + bus13_seg1[3] + m.v13_5[1,t]*A >= m.t13_5[1,t]
m.bus13_5_u1 = Constraint(m.PERIOD, rule = bus13_5_u1)

def bus13_5_l2(m,t):
    return bus13_seg2[0]*100*m.Pg[18,t]*m.ug[18,t] + bus13_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2[2]* m.inertia_constant[18]*m.genD_Pmax[18]*m.ug[18,t]) + bus13_seg2[3] <= m.t13_5[1,t]
m.bus13_5_l2 = Constraint(m.PERIOD,rule = bus13_5_l2)

def bus13_5_u2(m,t):
    return bus13_seg2[0]*100*m.Pg[18,t]*m.ug[18,t] + bus13_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2[2]* m.inertia_constant[18]*m.genD_Pmax[18]*m.ug[18,t]) + bus13_seg2[3] + (1 - m.v13_5[1,t])*A >= m.t13_5[1,t]
m.bus13_5_u2 = Constraint(m.PERIOD,rule = bus13_5_u2)

def bus13_5_t12(m,t):
    return m.t13_5[1,t] <= m.t13_5[2,t]
m.bus13_5_t12 = Constraint(m.PERIOD,rule = bus13_5_t12)

def bus13_5_t21(m,t):
    return m.t13_5[2,t]<=m.t13_5[1,t] + m.v13_5[2,t]*A
m.bus13_5_t21 = Constraint(m.PERIOD,rule = bus13_5_t21)

def bus13_5_l3(m,t):
    return bus13_seg3[0]*100*m.Pg[18,t]*m.ug[18,t] + bus13_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3[2]* m.inertia_constant[18]*m.genD_Pmax[18]*m.ug[18,t]) + bus13_seg3[3] <= m.t13_5[2,t]
m.bus13_5_l3 = Constraint(m.PERIOD,rule = bus13_5_l3)

def bus13_5_u3(m,t):
    return bus13_seg3[0]*100*m.Pg[18,t]*m.ug[18,t] + bus13_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3[2]* m.inertia_constant[18]*m.genD_Pmax[18]*m.ug[18,t]) + bus13_seg3[3] + (1 - m.v13_5[2,t])*A >= m.t13_5[2,t]
m.bus13_5_u3 = Constraint(m.PERIOD,rule = bus13_5_u3)

def bus13_5_t23(m,t):
    return m.t13_5[2,t] <= m.t13_5[3,t]
m.bus13_5_t23 = Constraint(m.PERIOD,rule = bus13_5_t23)

def bus13_5_t32(m,t):
    return m.t13_5[3,t]<=m.t13_5[2,t] + m.v13_5[3,t]*A
m.bus13_5_t32 = Constraint(m.PERIOD,rule = bus13_5_t32)

def bus13_5_l4(m,t):
    return bus13_seg4[0]*100*m.Pg[18,t]*m.ug[18,t] + bus13_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4[2]* m.inertia_constant[18]*m.genD_Pmax[18]*m.ug[18,t]) + bus13_seg4[3] <= m.t13_5[3,t]
m.bus13_5_l4 = Constraint(m.PERIOD,rule = bus13_5_l4)

def bus13_5_u4(m,t):
    return bus13_seg4[0]*100*m.Pg[18,t]*m.ug[18,t] + bus13_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4[2]* m.inertia_constant[18]*m.genD_Pmax[18]*m.ug[18,t]) + bus13_seg4[3] + (1 - m.v13_5[3,t])*A >= m.t13_5[3,t]
m.bus13_5_u4 = Constraint(m.PERIOD,rule = bus13_5_u4)

def bus13_6_R(m,t):
    return m.t13_6[3,t] <= RoCoF
m.bus13_6_R = Constraint(m.PERIOD, rule = bus13_6_R)

def bus13_6_l1(m,t):
    return bus13_seg1[0]*100*m.Pg[19,t]*m.ug[19,t] + bus13_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1[2]* m.inertia_constant[19]*m.genD_Pmax[19]*m.ug[19,t]) + bus13_seg1[3] <= m.t13_6[1,t]
m.bus13_6_l1 = Constraint(m.PERIOD, rule = bus13_6_l1)

def bus13_6_u1(m,t):
    return bus13_seg1[0]*100*m.Pg[19,t]*m.ug[19,t] + bus13_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1[2]* m.inertia_constant[19]*m.genD_Pmax[19]*m.ug[19,t]) + bus13_seg1[3] + m.v13_6[1,t]*A >= m.t13_6[1,t]
m.bus13_6_u1 = Constraint(m.PERIOD, rule = bus13_6_u1)

def bus13_6_l2(m,t):
    return bus13_seg2[0]*100*m.Pg[19,t]*m.ug[19,t] + bus13_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2[2]* m.inertia_constant[19]*m.genD_Pmax[19]*m.ug[19,t]) + bus13_seg2[3] <= m.t13_6[1,t]
m.bus13_6_l2 = Constraint(m.PERIOD,rule = bus13_6_l2)

def bus13_6_u2(m,t):
    return bus13_seg2[0]*100*m.Pg[19,t]*m.ug[19,t] + bus13_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2[2]* m.inertia_constant[19]*m.genD_Pmax[19]*m.ug[19,t]) + bus13_seg2[3] + (1 - m.v13_6[1,t])*A >= m.t13_6[1,t]
m.bus13_6_u2 = Constraint(m.PERIOD,rule = bus13_6_u2)

def bus13_6_t12(m,t):
    return m.t13_6[1,t] <= m.t13_6[2,t]
m.bus13_6_t12 = Constraint(m.PERIOD,rule = bus13_6_t12)

def bus13_6_t21(m,t):
    return m.t13_6[2,t]<=m.t13_6[1,t] + m.v13_6[2,t]*A
m.bus13_6_t21 = Constraint(m.PERIOD,rule = bus13_6_t21)

def bus13_6_l3(m,t):
    return bus13_seg3[0]*100*m.Pg[19,t]*m.ug[19,t] + bus13_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3[2]* m.inertia_constant[19]*m.genD_Pmax[19]*m.ug[19,t]) + bus13_seg3[3] <= m.t13_6[2,t]
m.bus13_6_l3 = Constraint(m.PERIOD,rule = bus13_6_l3)

def bus13_6_u3(m,t):
    return bus13_seg3[0]*100*m.Pg[19,t]*m.ug[19,t] + bus13_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3[2]* m.inertia_constant[19]*m.genD_Pmax[19]*m.ug[19,t]) + bus13_seg3[3] + (1 - m.v13_6[2,t])*A >= m.t13_6[2,t]
m.bus13_6_u3 = Constraint(m.PERIOD,rule = bus13_6_u3)

def bus13_6_t23(m,t):
    return m.t13_6[2,t] <= m.t13_6[3,t]
m.bus13_6_t23 = Constraint(m.PERIOD,rule = bus13_6_t23)

def bus13_6_t32(m,t):
    return m.t13_6[3,t]<=m.t13_6[2,t] + m.v13_6[3,t]*A
m.bus13_6_t32 = Constraint(m.PERIOD,rule = bus13_6_t32)

def bus13_6_l4(m,t):
    return bus13_seg4[0]*100*m.Pg[19,t]*m.ug[19,t] + bus13_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4[2]* m.inertia_constant[19]*m.genD_Pmax[19]*m.ug[19,t]) + bus13_seg4[3] <= m.t13_6[3,t]
m.bus13_6_l4 = Constraint(m.PERIOD,rule = bus13_6_l4)

def bus13_6_u4(m,t):
    return bus13_seg4[0]*100*m.Pg[19,t]*m.ug[19,t] + bus13_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4[2]* m.inertia_constant[19]*m.genD_Pmax[19]*m.ug[19,t]) + bus13_seg4[3] + (1 - m.v13_6[3,t])*A >= m.t13_6[3,t]
m.bus13_6_u4 = Constraint(m.PERIOD,rule = bus13_6_u4)


# Node 15 locational RoCoF constraints

def bus15_R(m,g,t):
    if g >= 20:
        if g <= 24:
            return m.t15[3,t] <= RoCoF
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_R = Constraint(m.GEND, m.PERIOD,rule = bus15_R)

def bus15_l1(m,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg1[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg1[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg1[3]<= m.t15[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_l1 = Constraint(m.GEND,m.PERIOD,rule = bus15_l1)

def bus15_u1(m,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg1[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg1[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg1[3] + m.v15[1,t]*A >= m.t15[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_u1 = Constraint(m.GEND, m.PERIOD, rule = bus15_u1)

def bus15_l2(m,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg2[3] <= m.t15[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_l2 = Constraint(m.GEND, m.PERIOD,rule = bus15_l2)

def bus15_u2(m,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg2[3] + ( 1 - m.v15[1, t]) * A >= m.t15[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_u2 = Constraint(m.GEND, m.PERIOD, rule = bus15_u2)

def bus15_t12(m,g,t):
    if g >= 20:
        if g <= 24:
            return m.t15[1,t] <= m.t15[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_t12 = Constraint(m.GEND, m.PERIOD,rule = bus15_t12)

def bus15_t21(m,g,t):
    if g >= 20:
        if g <= 24:
            return m.t15[2,t]<=m.t15[1,t] + m.v15[2,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_t21 = Constraint(m.GEND,m.PERIOD, rule = bus15_t21)

def bus15_l3(m,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg3[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg3[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg3[3] <= m.t15[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_l3 = Constraint(m.GEND,m.PERIOD, rule = bus15_l3)

def bus15_u3(m,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg3[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg3[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg3[3] + (1 - m.v15[2,t])*A >= m.t15[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_u3 = Constraint(m.GEND, m.PERIOD,rule = bus15_u3)

def bus15_t23(m,g,t):
    if g >= 20:
        if g <= 24:
            return m.t15[2,t] <= m.t15[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_t23 = Constraint(m.GEND,m.PERIOD, rule = bus15_t23)

def bus15_t32(m,g,t):
    if g >= 20:
        if g <= 24:
            return m.t15[3,t]<=m.t15[2,t] + m.v15[3,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_t32 = Constraint(m.GEND, m.PERIOD,rule = bus15_t32)

def bus15_l4(m,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg4[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg4[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg4[3] <= m.t15[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_l4 = Constraint(m.GEND,m.PERIOD,rule = bus15_l4)

def bus15_u4(m,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg4[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg4[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg4[3] + (1 - m.v15[3,t])*A >= m.t15[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_u4 = Constraint(m.GEND,m.PERIOD,rule = bus15_u4)

# Node 15_1 constraint
def bus15_1_R(m,g,t):
    if g >= 25:
        if g <= 26:
            return m.t15_1[3,t] <= RoCoF
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_R = Constraint(m.GEND, m.PERIOD,rule = bus15_1_R)

def bus15_1_l1(m,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg1[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg1[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg1[3] <= m.t15_1[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_l1 = Constraint(m.GEND,m.PERIOD,rule = bus15_1_l1)

def bus15_1_u1(m,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg1[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg1[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg1[3]+ m.v15_1[1,t]*A >= m.t15_1[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_u1 = Constraint(m.GEND, m.PERIOD, rule = bus15_1_u1)

def bus15_1_l2(m,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg2[3] <= m.t15_1[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_l2 = Constraint(m.GEND, m.PERIOD,rule = bus15_1_l2)

def bus15_1_u2(m,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg2[3] + ( 1 - m.v15_1[1, t]) * A >= m.t15_1[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_u2 = Constraint(m.GEND, m.PERIOD, rule = bus15_1_u2)

def bus15_1_t12(m,g,t):
    if g >= 25:
        if g <= 26:
            return m.t15_1[1,t] <= m.t15_1[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_t12 = Constraint(m.GEND, m.PERIOD,rule = bus15_1_t12)

def bus15_1_t21(m,g,t):
    if g >= 25:
        if g <= 26:
            return m.t15_1[2,t]<=m.t15_1[1,t] + m.v15_1[2,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_t21 = Constraint(m.GEND,m.PERIOD, rule = bus15_1_t21)

def bus15_1_l3(m,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg3[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg3[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg3[3] <= m.t15_1[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_l3 = Constraint(m.GEND,m.PERIOD, rule = bus15_1_l3)

def bus15_1_u3(m,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg3[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg3[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg3[3] + (1 - m.v15_1[2,t])*A >= m.t15_1[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_u3 = Constraint(m.GEND, m.PERIOD,rule = bus15_1_u3)

def bus15_1_t23(m,g,t):
    if g >= 25:
        if g <= 26:
            return m.t15_1[2,t] <= m.t15_1[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_t23 = Constraint(m.GEND,m.PERIOD, rule = bus15_1_t23)

def bus15_1_t32(m,g,t):
    if g >= 25:
        if g <= 26:
            return m.t15_1[3,t]<=m.t15_1[2,t] + m.v15_1[3,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_t32 = Constraint(m.GEND, m.PERIOD,rule = bus15_1_t32)

def bus15_1_l4(m,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg4[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg4[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg4[3] <= m.t15_1[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_l4 = Constraint(m.GEND,m.PERIOD,rule = bus15_1_l4)

def bus15_1_u4(m,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg4[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg4[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg4[3] + (1 - m.v15_1[3,t])*A >= m.t15_1[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_u4 = Constraint(m.GEND,m.PERIOD,rule = bus15_1_u4)

# Node 16 locational RoCoF constraints

def bus16_R(m,g,t):
    if g >= 27:
        if g <= 29:
            return m.t16[3,t] <= RoCoF
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_R = Constraint(m.GEND, m.PERIOD,rule = bus16_R)

def bus16_l1(m,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg1[0]*100*m.Pg[g,t]*m.ug[g,t] + bus16_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus16_seg1[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus16_seg1[3] <= m.t16[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_l1 = Constraint(m.GEND,m.PERIOD,rule = bus16_l1)

def bus16_u1(m,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg1[0]*100*m.Pg[g,t]*m.ug[g,t] + bus16_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus16_seg1[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus16_seg1[3]+ m.v16[1,t]*A >= m.t16[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_u1 = Constraint(m.GEND, m.PERIOD, rule = bus16_u1)

def bus16_l2(m,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus16_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus16_seg2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus16_seg2[3]<= m.t16[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_l2 = Constraint(m.GEND, m.PERIOD,rule = bus16_l2)

def bus16_u2(m,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus16_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus16_seg2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus16_seg2[3] + ( 1 - m.v16[1, t]) * A >= m.t16[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_u2 = Constraint(m.GEND, m.PERIOD, rule = bus16_u2)

def bus16_t12(m,g,t):
    if g >= 27:
        if g <= 29:
            return m.t16[1,t] <= m.t16[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_t12 = Constraint(m.GEND, m.PERIOD,rule = bus16_t12)

def bus16_t21(m,g,t):
    if g >= 27:
        if g <= 29:
            return m.t16[2,t]<=m.t16[1,t] + m.v16[2,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_t21 = Constraint(m.GEND,m.PERIOD, rule = bus16_t21)

def bus16_l3(m,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg3[0]*100*m.Pg[g,t]*m.ug[g,t] + bus16_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus16_seg3[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus16_seg3[3] <= m.t16[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_l3 = Constraint(m.GEND,m.PERIOD, rule = bus16_l3)

def bus16_u3(m,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg3[0]*100*m.Pg[g,t]*m.ug[g,t] + bus16_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus16_seg3[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus16_seg3[3] + (1 - m.v16[2,t])*A >= m.t16[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_u3 = Constraint(m.GEND, m.PERIOD,rule = bus16_u3)

def bus16_t23(m,g,t):
    if g >= 27:
        if g <= 29:
            return m.t16[2,t] <= m.t16[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_t23 = Constraint(m.GEND,m.PERIOD, rule = bus16_t23)

def bus16_t32(m,g,t):
    if g >= 27:
        if g <= 29:
            return m.t16[3,t]<=m.t16[2,t] + m.v16[3,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_t32 = Constraint(m.GEND, m.PERIOD,rule = bus16_t32)

def bus16_l4(m,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg4[0]*100*m.Pg[g,t]*m.ug[g,t] + bus16_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus16_seg4[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus16_seg4[3] <= m.t16[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_l4 = Constraint(m.GEND,m.PERIOD,rule = bus16_l4)

def bus16_u4(m,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg4[0]*100*m.Pg[g,t]*m.ug[g,t] + bus16_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus16_seg4[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus16_seg4[3] + (1 - m.v16[3,t])*A >= m.t16[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_u4 = Constraint(m.GEND,m.PERIOD,rule = bus16_u4)

# Node 18 locational RoCoF constraints

def bus18_R(m,t):
    return m.t18[3,t] <= RoCoF
m.bus18_R = Constraint(m.PERIOD, rule = bus18_R)

def bus18_l1(m,t):
    return bus18_seg1[0]*100*m.Pg[30,t]*m.ug[30,t] + bus18_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus18_seg1[2]* m.inertia_constant[30]*m.genD_Pmax[30]*m.ug[30,t]) + bus18_seg1[3] <= m.t18[1,t]
m.bus18_l1 = Constraint(m.PERIOD, rule = bus18_l1)

def bus18_u1(m,t):
    return bus18_seg1[0]*100*m.Pg[30,t]*m.ug[30,t] + bus18_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus18_seg1[2]* m.inertia_constant[30]*m.genD_Pmax[30]*m.ug[30,t]) + bus18_seg1[3] + m.v18[1,t]*A >= m.t18[1,t]
m.bus18_u1 = Constraint(m.PERIOD, rule = bus18_u1)

def bus18_l2(m,t):
    return bus18_seg2[0]*100*m.Pg[30,t]*m.ug[30,t] + bus18_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus18_seg2[2]* m.inertia_constant[30]*m.genD_Pmax[30]*m.ug[30,t]) + bus18_seg2[3] <= m.t18[1,t]
m.bus18_l2 = Constraint(m.PERIOD,rule = bus18_l2)

def bus18_u2(m,t):
    return bus18_seg2[0]*100*m.Pg[30,t]*m.ug[30,t] + bus18_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus18_seg2[2]* m.inertia_constant[30]*m.genD_Pmax[30]*m.ug[30,t]) + bus18_seg2[3]+ (1 - m.v18[1,t])*A >= m.t18[1,t]
m.bus18_u2 = Constraint(m.PERIOD,rule = bus18_u2)

def bus18_t12(m,t):
    return m.t18[1,t] <= m.t18[2,t]
m.bus18_t12 = Constraint(m.PERIOD,rule = bus18_t12)

def bus18_t21(m,t):
    return m.t18[2,t]<=m.t18[1,t] + m.v18[2,t]*A
m.bus18_t21 = Constraint(m.PERIOD,rule = bus18_t21)

def bus18_l3(m,t):
    return bus18_seg3[0]*100*m.Pg[30,t]*m.ug[30,t] + bus18_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus18_seg3[2]* m.inertia_constant[30]*m.genD_Pmax[30]*m.ug[30,t]) + bus18_seg3[3]<= m.t18[2,t]
m.bus18_l3 = Constraint(m.PERIOD,rule = bus18_l3)

def bus18_u3(m,t):
    return bus18_seg3[0]*100*m.Pg[30,t]*m.ug[30,t] + bus18_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus18_seg3[2]* m.inertia_constant[30]*m.genD_Pmax[30]*m.ug[30,t]) + bus18_seg3[3] + (1 - m.v18[2,t])*A >= m.t18[2,t]
m.bus18_u3 = Constraint(m.PERIOD,rule = bus18_u3)

def bus18_t23(m,t):
    return m.t18[2,t] <= m.t18[3,t]
m.bus18_t23 = Constraint(m.PERIOD,rule = bus18_t23)

def bus18_t32(m,t):
    return m.t18[3,t]<=m.t18[2,t] + m.v18[3,t]*A
m.bus18_t32 = Constraint(m.PERIOD,rule = bus18_t32)

def bus18_l4(m,t):
    return bus18_seg4[0]*100*m.Pg[30,t]*m.ug[30,t] + bus18_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus18_seg4[2]* m.inertia_constant[30]*m.genD_Pmax[30]*m.ug[30,t]) + bus18_seg4[3] <= m.t18[3,t]
m.bus18_l4 = Constraint(m.PERIOD,rule = bus18_l4)

def bus18_u4(m,t):
    return bus18_seg4[0]*100*m.Pg[30,t]*m.ug[30,t] + bus18_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus18_seg4[2]* m.inertia_constant[30]*m.genD_Pmax[30]*m.ug[30,t]) + bus18_seg4[3] + (1 - m.v18[3,t])*A >= m.t18[3,t]
m.bus18_u4 = Constraint(m.PERIOD,rule = bus18_u4)

# Node 21 locational RoCoF constraints

def bus21_R(m,g,t):
    if g >= 31:
        if g <= 35:
            return m.t21[3,t] <= RoCoF
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_R = Constraint(m.GEND, m.PERIOD,rule = bus21_R)

def bus21_l1(m,g,t):
    if g >= 31:
        if g <= 35:
            return 0.0027*100*m.Pg[g,t]*m.ug[g,t]- 0.000001*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) - m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) <= m.t21[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_l1 = Constraint(m.GEND,m.PERIOD,rule = bus21_l1)

def bus21_u1(m,g,t):
    if g >= 31:
        if g <= 35:
            return 0.0027*100*m.Pg[g,t]*m.ug[g,t]- 0.000001*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) - m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) + m.v21[1,t]*A >= m.t21[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_u1 = Constraint(m.GEND, m.PERIOD, rule = bus21_u1)

def bus21_l2(m,g,t):
    if g >= 31:
        if g <= 35:
            return 0.0013*100*m.Pg[g,t]*m.ug[g,t]- 0.000001*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) - m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t])  + 0.3457 <= m.t21[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_l2 = Constraint(m.GEND, m.PERIOD,rule = bus21_l2)

def bus21_u2(m,g,t):
    if g >= 31:
        if g <= 35:
            return 0.0013 * 100 * m.Pg[g, t] * m.ug[g, t] - 0.000001*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) - m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t])+ 0.3457 + ( 1 - m.v21[1, t]) * A >= m.t21[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_u2 = Constraint(m.GEND, m.PERIOD, rule = bus21_u2)

def bus21_t12(m,g,t):
    if g >= 31:
        if g <= 35:
            return m.t21[1,t] <= m.t21[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_t12 = Constraint(m.GEND, m.PERIOD,rule = bus21_t12)

def bus21_t21(m,g,t):
    if g >= 31:
        if g <= 35:
            return m.t21[2,t]<=m.t21[1,t] + m.v21[2,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_t21 = Constraint(m.GEND,m.PERIOD, rule = bus21_t21)

def bus21_l3(m,g,t):
    if g >= 31:
        if g <= 35:
            return 0.0027*100*m.Pg[g,t]*m.ug[g,t] - 0.000001*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) - m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) <= m.t21[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_l3 = Constraint(m.GEND,m.PERIOD, rule = bus21_l3)

def bus21_u3(m,g,t):
    if g >= 31:
        if g <= 35:
            return 0.0027*100*m.Pg[g,t]*m.ug[g,t] - 0.000001*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) - m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) + (1 - m.v21[2,t])*A >= m.t21[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_u3 = Constraint(m.GEND, m.PERIOD,rule = bus21_u3)

def bus21_t23(m,g,t):
    if g >= 31:
        if g <= 35:
            return m.t21[2,t] <= m.t21[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_t23 = Constraint(m.GEND,m.PERIOD, rule = bus21_t23)

def bus21_t32(m,g,t):
    if g >= 31:
        if g <= 35:
            return m.t21[3,t]<=m.t21[2,t] + m.v21[3,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_t32 = Constraint(m.GEND, m.PERIOD,rule = bus21_t32)

def bus21_l4(m,g,t):
    if g >= 31:
        if g <= 35:
            return 0 <= m.t21[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_l4 = Constraint(m.GEND,m.PERIOD,rule = bus21_l4)

def bus21_u4(m,g,t):
    if g >= 31:
        if g <= 35:
            return (1 - m.v21[3,t])*A >= m.t21[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_u4 = Constraint(m.GEND,m.PERIOD,rule = bus21_u4)


# Node 22 constraints
def bus22_R(m,t):
    return m.t22[3,t] <= RoCoF
m.bus22_R = Constraint(m.PERIOD, rule = bus22_R)

def bus22_l1(m,t):
    return bus22_seg1[0]*100*m.Pg[36,t]*m.ug[36,t] + bus22_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus22_seg1[2]* m.inertia_constant[36]*m.genD_Pmax[36]*m.ug[36,t]) + bus22_seg1[3] <= m.t22[1,t]
m.bus22_l1 = Constraint(m.PERIOD, rule = bus22_l1)

def bus22_u1(m,t):
    return bus22_seg1[0]*100*m.Pg[36,t]*m.ug[36,t] + bus22_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus22_seg1[2]* m.inertia_constant[36]*m.genD_Pmax[36]*m.ug[36,t]) + bus22_seg1[3] + m.v22[1,t]*A >= m.t22[1,t]
m.bus22_u1 = Constraint(m.PERIOD, rule = bus22_u1)

def bus22_l2(m,t):
    return bus22_seg2[0]*100*m.Pg[36,t]*m.ug[36,t] + bus22_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus22_seg2[2]* m.inertia_constant[36]*m.genD_Pmax[36]*m.ug[36,t]) + bus22_seg2[3] <= m.t22[1,t]
m.bus22_l2 = Constraint(m.PERIOD,rule = bus22_l2)

def bus22_u2(m,t):
    return bus22_seg2[0]*100*m.Pg[36,t]*m.ug[36,t] + bus22_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus22_seg2[2]* m.inertia_constant[36]*m.genD_Pmax[36]*m.ug[36,t]) + bus22_seg2[3] + (1 - m.v22[1,t])*A >= m.t22[1,t]
m.bus22_u2 = Constraint(m.PERIOD,rule = bus22_u2)

def bus22_t12(m,t):
    return m.t22[1,t] <= m.t22[2,t]
m.bus22_t12 = Constraint(m.PERIOD,rule = bus22_t12)

def bus22_t21(m,t):
    return m.t22[2,t]<=m.t22[1,t] + m.v22[2,t]*A
m.bus22_t21 = Constraint(m.PERIOD,rule = bus22_t21)

def bus22_l3(m,t):
    return bus22_seg3[0]*100*m.Pg[36,t]*m.ug[36,t] + bus22_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus22_seg3[2]* m.inertia_constant[36]*m.genD_Pmax[36]*m.ug[36,t]) + bus22_seg3[3] <= m.t22[2,t]
m.bus22_l3 = Constraint(m.PERIOD,rule = bus22_l3)

def bus22_u3(m,t):
    return bus22_seg3[0]*100*m.Pg[36,t]*m.ug[36,t] + bus22_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus22_seg3[2]* m.inertia_constant[36]*m.genD_Pmax[36]*m.ug[36,t]) + bus22_seg3[3] + (1 - m.v22[2,t])*A >= m.t22[2,t]
m.bus22_u3 = Constraint(m.PERIOD,rule = bus22_u3)

def bus22_t23(m,t):
    return m.t22[2,t] <= m.t22[3,t]
m.bus22_t23 = Constraint(m.PERIOD,rule = bus22_t23)

def bus22_t32(m,t):
    return m.t22[3,t]<=m.t22[2,t] + m.v22[3,t]*A
m.bus22_t32 = Constraint(m.PERIOD,rule = bus22_t32)

def bus22_l4(m,t):
    return bus22_seg4[0]*100*m.Pg[36,t]*m.ug[36,t] + bus22_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus22_seg4[2]* m.inertia_constant[36]*m.genD_Pmax[36]*m.ug[36,t]) + bus22_seg4[3] <= m.t22[3,t]
m.bus22_l4 = Constraint(m.PERIOD,rule = bus22_l4)

def bus22_u4(m,t):
    return bus22_seg4[0]*100*m.Pg[36,t]*m.ug[36,t] + bus22_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus22_seg4[2]* m.inertia_constant[36]*m.genD_Pmax[36]*m.ug[36,t]) + bus22_seg4[3] + (1 - m.v22[3,t])*A >= m.t22[3,t]
m.bus22_u4 = Constraint(m.PERIOD,rule = bus22_u4)


# Node 23 locational RoCoF constraints

def bus23_R(m,g,t):
    if g >= 39:
        if g <= 41:
            return m.t23[3,t] <= RoCoF
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_R = Constraint(m.GEND, m.PERIOD,rule = bus23_R)

def bus23_l1(m,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg1[0]*100*m.Pg[g,t]*m.ug[g,t] + bus23_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus23_seg1[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus23_seg1[3]  <= m.t23[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_l1 = Constraint(m.GEND,m.PERIOD,rule = bus23_l1)

def bus23_u1(m,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg1[0]*100*m.Pg[g,t]*m.ug[g,t] + bus23_seg1[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus23_seg1[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus23_seg1[3]  + m.v23[1,t]*A >= m.t23[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_u1 = Constraint(m.GEND, m.PERIOD, rule = bus23_u1)

def bus23_l2(m,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus23_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus23_seg2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus23_seg2[3]  <= m.t23[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_l2 = Constraint(m.GEND, m.PERIOD,rule = bus23_l2)

def bus23_u2(m,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus23_seg2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus23_seg2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus23_seg2[3]  + ( 1 - m.v23[1, t]) * A >= m.t23[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_u2 = Constraint(m.GEND, m.PERIOD, rule = bus23_u2)

def bus23_t12(m,g,t):
    if g >= 39:
        if g <= 41:
            return m.t23[1,t] <= m.t23[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_t12 = Constraint(m.GEND, m.PERIOD,rule = bus23_t12)

def bus23_t21(m,g,t):
    if g >= 39:
        if g <= 41:
            return m.t23[2,t]<=m.t23[1,t] + m.v23[2,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_t21 = Constraint(m.GEND,m.PERIOD, rule = bus23_t21)

def bus23_l3(m,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg3[0]*100*m.Pg[g,t]*m.ug[g,t] + bus23_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus23_seg3[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus23_seg3[3]  <= m.t23[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_l3 = Constraint(m.GEND,m.PERIOD, rule = bus23_l3)

def bus23_u3(m,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg3[0]*100*m.Pg[g,t]*m.ug[g,t] + bus23_seg3[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus23_seg3[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus23_seg3[3]  + (1 - m.v23[2,t])*A >= m.t23[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_u3 = Constraint(m.GEND, m.PERIOD,rule = bus23_u3)

def bus23_t23(m,g,t):
    if g >= 39:
        if g <= 41:
            return m.t23[2,t] <= m.t23[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_t23 = Constraint(m.GEND,m.PERIOD, rule = bus23_t23)

def bus23_t32(m,g,t):
    if g >= 39:
        if g <= 41:
            return m.t23[3,t]<=m.t23[2,t] + m.v23[3,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_t32 = Constraint(m.GEND, m.PERIOD,rule = bus23_t32)

def bus23_l4(m,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg4[0]*100*m.Pg[g,t]*m.ug[g,t] + bus23_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus23_seg4[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus23_seg4[3]  <= m.t23[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_l4 = Constraint(m.GEND,m.PERIOD,rule = bus23_l4)

def bus23_u4(m,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg4[0]*100*m.Pg[g,t]*m.ug[g,t] + bus23_seg4[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus23_seg4[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus23_seg4[3]  + (1 - m.v23[3,t])*A >= m.t23[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_u4 = Constraint(m.GEND,m.PERIOD,rule = bus23_u4)


def bus01_R_t2(m,t):
    return m.t01_t2[3,t] <= RoCoF
m.bus01_R_t2 = Constraint(m.PERIOD,rule = bus01_R_t2)

def bus01_l1_t2(m,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg1_t2[0]*100*m.Pg[g,t]*m.ug[g,t]+ bus1_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND)) + bus1_seg1_t2[2] * m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t] + bus1_seg1_t2[3] <= m.t01_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus01_l1_t2 = Constraint(m.GEND,m.PERIOD,rule = bus01_l1_t2)

def bus01_u1_t2(m,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg1_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus1_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus1_seg1_t2[2] * m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) + bus1_seg1_t2[3] + m.v01_t2[1,t]*A >= m.t01_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus01_u1_t2 = Constraint(m.GEND, m.PERIOD, rule = bus01_u1_t2)

def bus01_l2_t2(m,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg2_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus1_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND)) + bus1_seg2_t2[2] * m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t] +  bus1_seg2_t2[3] <= m.t01_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus01_l2_t2 = Constraint(m.GEND, m.PERIOD,rule = bus01_l2_t2)

def bus01_u2_t2(m,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg2_t2[0]*100*m.Pg[g,t]*m.ug[g,t] +  bus1_seg2_t2[1]/(fn*np.pi)*(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND)) + bus1_seg2_t2[2] * m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]+  bus1_seg2_t2[3] + ( 1 - m.v01_t2[1, t]) * A >= m.t01_t2[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus01_u2_t2 = Constraint(m.GEND, m.PERIOD, rule = bus01_u2_t2)

def bus01_t12_t2(m,t):
    return m.t01_t2[1,t] <= m.t01_t2[2,t]
m.bus01_t12_t2 = Constraint( m.PERIOD,rule = bus01_t12_t2)

def bus01_t21_t2(m,t):
    return m.t01_t2[2,t]<=m.t01_t2[1,t] + m.v01_t2[2,t]*A
m.bus01_t21_t2 = Constraint(m.PERIOD, rule = bus01_t21_t2)

def bus01_l3_t2(m,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg3_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus1_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus1_seg3_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus1_seg3_t2[3]<= m.t01_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus01_l3_t2 = Constraint(m.GEND,m.PERIOD, rule = bus01_l3_t2)

def bus01_u3_t2(m,g,t):
    if g >= 1:
        if g <= 4:
            return bus1_seg3_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus1_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND)) + bus1_seg3_t2[2]* m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]  + bus1_seg3_t2[3] + (1 - m.v01_t2[2,t])*A >= m.t01_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus01_u3_t2 = Constraint(m.GEND, m.PERIOD,rule = bus01_u3_t2)

def bus01_t23_t2(m,t):
    return m.t01_t2[2,t] <= m.t01_t2[3,t]
m.bus01_t23_t2 = Constraint(m.PERIOD, rule = bus01_t23_t2)

def bus01_t32_t2(m,t):
    return m.t01_t2[3,t]<=m.t01_t2[2,t] + m.v01_t2[3,t]*A
m.bus01_t32_t2 = Constraint(m.PERIOD,rule = bus01_t32_t2)

def bus01_l4_t2(m,g,t):
    if g >= 1:
        if g <= 4:
             return bus1_seg4_t2[0]*m.Pg[g,t]*m.ug[g,t] + bus1_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND)) + bus1_seg4_t2[2]* m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t] + bus1_seg4_t2[3] <= m.t01_t2[3,t]
        else:
             return Constraint.Skip
    else:
        return Constraint.Skip
m.bus01_l4_t2 = Constraint(m.GEND,m.PERIOD,rule = bus01_l4_t2)

def bus01_u4_t2(m,g,t):
    if g >= 1:
        if g <= 4:
             return bus1_seg4_t2[0]*m.Pg[g,t]*m.ug[g,t] + bus1_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND)) + bus1_seg4_t2[2]* m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t] + bus1_seg4_t2[3] +(1 - m.v01_t2[3,t])*A >= m.t01_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus01_u4_t2 = Constraint(m.GEND,m.PERIOD,rule = bus01_u4_t2)

# Node 02 locational RoCoF constraints

def bus02_1_R_t2(m,t):
    return m.t02_1_t2[3,t] <= RoCoF
m.bus02_1_R_t2 = Constraint(m.PERIOD, rule = bus02_1_R_t2)

def bus02_1_l1_t2(m,t):
    return bus2_seg1_t2[0]*100*m.Pg[5,t]*m.ug[5,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1_t2[2] * m.inertia_constant[5]*m.genD_Pmax[5]*m.ug[5,t]) + bus2_seg1_t2[3] <= m.t02_1_t2[1,t]
m.bus02_1_l1_t2 = Constraint(m.PERIOD, rule = bus02_1_l1_t2)

def bus02_1_u1_t2(m,t):
    return bus2_seg1_t2[0]*100*m.Pg[5,t]*m.ug[5,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1_t2[2] * m.inertia_constant[5]*m.genD_Pmax[5]*m.ug[5,t]) + bus2_seg1_t2[3] + m.v02_1_t2[1,t]*A >= m.t02_1_t2[1,t]
m.bus02_1_u1_t2 = Constraint(m.PERIOD, rule = bus02_1_u1_t2)

def bus02_1_l2_t2(m,t):
    return bus2_seg2_t2[0]*100*m.Pg[5,t]*m.ug[5,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2_t2[2] * m.inertia_constant[5]*m.genD_Pmax[5]*m.ug[5,t]) + bus2_seg2_t2[3] <= m.t02_1_t2[1,t]
m.bus02_1_l2_t2 = Constraint(m.PERIOD,rule = bus02_1_l2_t2)

def bus02_1_u2_t2(m,t):
    return bus2_seg2_t2[0]*100*m.Pg[5,t]*m.ug[5,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2_t2[2] * m.inertia_constant[5]*m.genD_Pmax[5]*m.ug[5,t]) + bus2_seg2_t2[3] + (1 - m.v02_1_t2[1,t])*A >= m.t02_1_t2[1,t]
m.bus02_1_u2_t2 = Constraint(m.PERIOD,rule = bus02_1_u2_t2)

def bus02_1_t12_t2(m,t):
    return m.t02_1_t2[1,t] <= m.t02_1_t2[2,t]
m.bus02_1_t12_t2 = Constraint(m.PERIOD,rule = bus02_1_t12_t2)

def bus02_1_t21_t2(m,t):
    return m.t02_1_t2[2,t]<=m.t02_1_t2[1,t] + m.v02_1_t2[2,t]*A
m.bus02_1_t21_t2 = Constraint(m.PERIOD,rule = bus02_1_t21_t2)

def bus02_1_l3_t2(m,t):
    return bus2_seg3_t2[0]*100*m.Pg[5,t]*m.ug[5,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3_t2[2]* m.inertia_constant[5]*m.genD_Pmax[5]*m.ug[5,t]) + bus2_seg3_t2[3]  <= m.t02_1_t2[2,t]
m.bus02_1_l3_t2 = Constraint(m.PERIOD,rule = bus02_1_l3_t2)

def bus02_1_u3_t2(m,t):
    return bus2_seg3_t2[0]*100*m.Pg[5,t]*m.ug[5,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3_t2[2]* m.inertia_constant[5]*m.genD_Pmax[5]*m.ug[5,t]) + bus2_seg3_t2[3]   + (1 - m.v02_1_t2[2,t])*A >= m.t02_1_t2[2,t]
m.bus02_1_u3_t2 = Constraint(m.PERIOD,rule = bus02_1_u3_t2)

def bus02_1_t23_t2(m,t):
    return m.t02_1_t2[2,t] <= m.t02_1_t2[3,t]
m.bus02_1_t23_t2 = Constraint(m.PERIOD,rule = bus02_1_t23_t2)

def bus02_1_t32_t2(m,t):
    return m.t02_1_t2[3,t]<=m.t02_1_t2[2,t] + m.v02_1_t2[3,t]*A
m.bus02_1_t32_t2 = Constraint(m.PERIOD,rule = bus02_1_t32_t2)

def bus02_1_l4_t2(m,t):
    return bus2_seg4_t2[0]*100*m.Pg[5,t]*m.ug[5,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4_t2[2]* m.inertia_constant[5]*m.genD_Pmax[5]*m.ug[5,t]) + bus2_seg4_t2[3]  <= m.t02_1_t2[3,t]
m.bus02_1_l4_t2 = Constraint(m.PERIOD,rule = bus02_1_l4_t2)

def bus02_1_u4_t2(m,t):
    return bus2_seg4_t2[0]*100*m.Pg[5,t]*m.ug[5,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4_t2[2]* m.inertia_constant[5]*m.genD_Pmax[5]*m.ug[5,t]) + bus2_seg4_t2[3]+(1 - m.v02_1_t2[3,t])*A >= m.t02_1_t2[3,t]
m.bus02_1_u4_t2 = Constraint(m.PERIOD,rule = bus02_1_u4_t2)

def bus02_2_R_t2(m,t):
    return m.t02_2_t2[3,t] <= RoCoF
m.bus02_2_R_t2 = Constraint(m.PERIOD, rule = bus02_2_R_t2)

def bus02_2_l1_t2(m,t):
    return bus2_seg1_t2[0]*100*m.Pg[6,t]*m.ug[6,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1_t2[2]* m.inertia_constant[6]*m.genD_Pmax[6]*m.ug[6,t]) + bus2_seg1_t2[3]<= m.t02_2_t2[1,t]
m.bus02_2_l1_t2 = Constraint(m.PERIOD, rule = bus02_2_l1_t2)

def bus02_2_u1_t2(m,t):
    return bus2_seg1_t2[0]*100*m.Pg[6,t]*m.ug[6,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1_t2[2]* m.inertia_constant[6]*m.genD_Pmax[6]*m.ug[6,t]) + bus2_seg1_t2[3] + m.v02_2_t2[1,t]*A >= m.t02_2_t2[1,t]
m.bus02_2_u1_t2 = Constraint(m.PERIOD, rule = bus02_2_u1_t2)

def bus02_2_l2_t2(m,t):
    return bus2_seg2_t2[0]*100*m.Pg[6,t]*m.ug[6,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2_t2[2]* m.inertia_constant[6]*m.genD_Pmax[6]*m.ug[6,t]) + bus2_seg2_t2[3] <= m.t02_2_t2[1,t]
m.bus02_2_l2_t2 = Constraint(m.PERIOD,rule = bus02_2_l2_t2)

def bus02_2_u2_t2(m,t):
    return bus2_seg2_t2[0]*100*m.Pg[6,t]*m.ug[6,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2_t2[2]* m.inertia_constant[6]*m.genD_Pmax[6]*m.ug[6,t]) + bus2_seg2_t2[3]+ (1 - m.v02_2_t2[1,t])*A >= m.t02_2_t2[1,t]
m.bus02_2_u2_t2 = Constraint(m.PERIOD,rule = bus02_2_u2_t2)

def bus02_2_t12_t2(m,t):
    return m.t02_2_t2[1,t] <= m.t02_2_t2[2,t]
m.bus02_2_t12_t2 = Constraint(m.PERIOD,rule = bus02_2_t12_t2)

def bus02_2_t21_t2(m,t):
    return m.t02_2_t2[2,t]<=m.t02_2_t2[1,t] + m.v02_2_t2[2,t]*A
m.bus02_2_t21_t2 = Constraint(m.PERIOD,rule = bus02_2_t21_t2)

def bus02_2_l3_t2(m,t):
    return bus2_seg3_t2[0]*100*m.Pg[6,t]*m.ug[6,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3_t2[2]* m.inertia_constant[6]*m.genD_Pmax[6]*m.ug[6,t]) + bus2_seg3_t2[3] <= m.t02_2_t2[2,t]
m.bus02_2_l3_t2 = Constraint(m.PERIOD,rule = bus02_2_l3_t2)

def bus02_2_u3_t2(m,t):
    return bus2_seg3_t2[0]*100*m.Pg[6,t]*m.ug[6,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3_t2[2]* m.inertia_constant[6]*m.genD_Pmax[6]*m.ug[6,t]) + bus2_seg3_t2[3] + (1 - m.v02_2_t2[2,t])*A >= m.t02_2_t2[2,t]
m.bus02_2_u3_t2 = Constraint(m.PERIOD,rule = bus02_2_u3_t2)

def bus02_2_t23_t2(m,t):
    return m.t02_2_t2[2,t] <= m.t02_2_t2[3,t]
m.bus02_2_t23_t2 = Constraint(m.PERIOD,rule = bus02_2_t23_t2)

def bus02_2_t32_t2(m,t):
    return m.t02_2_t2[3,t]<=m.t02_2_t2[2,t] + m.v02_2_t2[3,t]*A
m.bus02_2_t32_t2 = Constraint(m.PERIOD,rule = bus02_2_t32_t2)

def bus02_2_l4_t2(m,t):
    return bus2_seg4_t2[0]*100*m.Pg[6,t]*m.ug[6,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4_t2[2]* m.inertia_constant[6]*m.genD_Pmax[6]*m.ug[6,t]) + bus2_seg4_t2[3] <= m.t02_2_t2[3,t]
m.bus02_2_l4_t2 = Constraint(m.PERIOD,rule = bus02_2_l4_t2)

def bus02_2_u4_t2(m,t):
    return bus2_seg4_t2[0]*100*m.Pg[6,t]*m.ug[6,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4_t2[2]* m.inertia_constant[6]*m.genD_Pmax[6]*m.ug[6,t]) + bus2_seg4_t2[3] + (1 - m.v02_2_t2[3,t])*A >= m.t02_2_t2[3,t]
m.bus02_2_u4_t2 = Constraint(m.PERIOD,rule = bus02_2_u4_t2)

def bus02_3_R_t2(m,t):
    return m.t02_3_t2[3,t] <= RoCoF
m.bus02_3_R_t2 = Constraint(m.PERIOD, rule = bus02_3_R_t2)

def bus02_3_l1_t2(m,t):
    return bus2_seg1_t2[0]*100*m.Pg[7,t]*m.ug[7,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1_t2[2]* m.inertia_constant[7]*m.genD_Pmax[7]*m.ug[7,t]) + bus2_seg1_t2[3] <= m.t02_3_t2[1,t]
m.bus02_3_l1_t2 = Constraint(m.PERIOD, rule = bus02_3_l1_t2)

def bus02_3_u1_t2(m,t):
    return bus2_seg1_t2[0]*100*m.Pg[7,t]*m.ug[7,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1_t2[2]* m.inertia_constant[7]*m.genD_Pmax[7]*m.ug[7,t]) + bus2_seg1_t2[3] + m.v02_3_t2[1,t]*A >= m.t02_3_t2[1,t]
m.bus02_3_u1_t2 = Constraint(m.PERIOD, rule = bus02_3_u1_t2)

def bus02_3_l2_t2(m,t):
    return bus2_seg2_t2[0]*100*m.Pg[7,t]*m.ug[7,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2_t2[2]* m.inertia_constant[7]*m.genD_Pmax[7]*m.ug[7,t]) + bus2_seg2_t2[3] <= m.t02_3_t2[1,t]
m.bus02_3_l2_t2 = Constraint(m.PERIOD,rule = bus02_3_l2_t2)

def bus02_3_u2_t2(m,t):
    return bus2_seg2_t2[0]*100*m.Pg[7,t]*m.ug[7,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2_t2[2]* m.inertia_constant[7]*m.genD_Pmax[7]*m.ug[7,t]) + bus2_seg2_t2[3]+ (1 - m.v02_3_t2[1,t])*A >= m.t02_3_t2[1,t]
m.bus02_3_u2_t2 = Constraint(m.PERIOD,rule = bus02_3_u2_t2)

def bus02_3_t12_t2(m,t):
    return m.t02_3_t2[1,t] <= m.t02_3_t2[2,t]
m.bus02_3_t12_t2 = Constraint(m.PERIOD,rule = bus02_3_t12_t2)

def bus02_3_t21_t2(m,t):
    return m.t02_3_t2[2,t]<=m.t02_3_t2[1,t] + m.v02_3_t2[2,t]*A
m.bus02_3_t21_t2 = Constraint(m.PERIOD,rule = bus02_3_t21_t2)

def bus02_3_l3_t2(m,t):
    return bus2_seg3_t2[0]*100*m.Pg[7,t]*m.ug[7,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3_t2[2]* m.inertia_constant[7]*m.genD_Pmax[7]*m.ug[7,t]) + bus2_seg3_t2[3]  <= m.t02_3_t2[2,t]
m.bus02_3_l3_t2 = Constraint(m.PERIOD,rule = bus02_3_l3_t2)

def bus02_3_u3_t2(m,t):
    return bus2_seg3_t2[0]*100*m.Pg[7,t]*m.ug[7,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3_t2[2]* m.inertia_constant[7]*m.genD_Pmax[7]*m.ug[7,t]) + bus2_seg3_t2[3] + (1 - m.v02_3_t2[2,t])*A >= m.t02_3_t2[2,t]
m.bus02_3_u3_t2 = Constraint(m.PERIOD,rule = bus02_3_u3_t2)

def bus02_3_t23_t2(m,t):
    return m.t02_3_t2[2,t] <= m.t02_3_t2[3,t]
m.bus02_3_t23_t2 = Constraint(m.PERIOD,rule = bus02_3_t23_t2)

def bus02_3_t32_t2(m,t):
    return m.t02_3_t2[3,t]<=m.t02_3_t2[2,t] + m.v02_3_t2[3,t]*A
m.bus02_3_t32_t2 = Constraint(m.PERIOD,rule = bus02_3_t32_t2)

def bus02_3_l4_t2(m,t):
    return bus2_seg4_t2[0]*100*m.Pg[7,t]*m.ug[7,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4_t2[2]* m.inertia_constant[7]*m.genD_Pmax[7]*m.ug[7,t]) + bus2_seg4_t2[3] <= m.t02_3_t2[3,t]
m.bus02_3_l4_t2 = Constraint(m.PERIOD,rule = bus02_3_l4_t2)

def bus02_3_u4_t2(m,t):
    return bus2_seg4_t2[0]*100*m.Pg[7,t]*m.ug[7,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4_t2[2]* m.inertia_constant[7]*m.genD_Pmax[7]*m.ug[7,t]) + bus2_seg4_t2[3] + (1 - m.v02_3_t2[3,t])*A >= m.t02_3_t2[3,t]
m.bus02_3_u4_t2 = Constraint(m.PERIOD,rule = bus02_3_u4_t2)

def bus02_4_R_t2(m,t):
    return m.t02_4_t2[3,t] <= RoCoF
m.bus02_4_R_t2 = Constraint(m.PERIOD, rule = bus02_4_R_t2)

def bus02_4_l1_t2(m,t):
    return bus2_seg1_t2[0]*100*m.Pg[8,t]*m.ug[8,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1_t2[2]* m.inertia_constant[8]*m.genD_Pmax[8]*m.ug[8,t]) + bus2_seg1_t2[3] <= m.t02_4_t2[1,t]
m.bus02_4_l1_t2 = Constraint(m.PERIOD, rule = bus02_4_l1_t2)

def bus02_4_u1_t2(m,t):
    return bus2_seg1_t2[0]*100*m.Pg[8,t]*m.ug[8,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1_t2[2]* m.inertia_constant[8]*m.genD_Pmax[8]*m.ug[8,t]) + bus2_seg1_t2[3] + m.v02_4_t2[1,t]*A >= m.t02_4_t2[1,t]
m.bus02_4_u1_t2 = Constraint(m.PERIOD, rule = bus02_4_u1_t2)

def bus02_4_l2_t2(m,t):
    return bus2_seg2_t2[0]*100*m.Pg[8,t]*m.ug[8,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2_t2[2]* m.inertia_constant[8]*m.genD_Pmax[8]*m.ug[8,t]) + bus2_seg2_t2[3] <= m.t02_4_t2[1,t]
m.bus02_4_l2_t2 = Constraint(m.PERIOD,rule = bus02_4_l2_t2)

def bus02_4_u2_t2(m,t):
    return bus2_seg2_t2[0]*100*m.Pg[8,t]*m.ug[8,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2_t2[2]* m.inertia_constant[8]*m.genD_Pmax[8]*m.ug[8,t]) + bus2_seg2_t2[3] + (1 - m.v02_4_t2[1,t])*A >= m.t02_4_t2[1,t]
m.bus02_4_u2_t2 = Constraint(m.PERIOD,rule = bus02_4_u2_t2)

def bus02_4_t12_t2(m,t):
    return m.t02_4_t2[1,t] <= m.t02_4_t2[2,t]
m.bus02_4_t12_t2 = Constraint(m.PERIOD,rule = bus02_4_t12_t2)

def bus02_4_t21_t2(m,t):
    return m.t02_4_t2[2,t]<=m.t02_4_t2[1,t] + m.v02_4_t2[2,t]*A
m.bus02_4_t21_t2 = Constraint(m.PERIOD,rule = bus02_4_t21_t2)

def bus02_4_l3_t2(m,t):
    return bus2_seg3_t2[0]*100*m.Pg[8,t]*m.ug[8,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3_t2[2]* m.inertia_constant[8]*m.genD_Pmax[8]*m.ug[8,t]) + bus2_seg3_t2[3]  <= m.t02_4_t2[2,t]
m.bus02_4_l3_t2 = Constraint(m.PERIOD,rule = bus02_4_l3_t2)

def bus02_4_u3_t2(m,t):
    return bus2_seg3_t2[0]*100*m.Pg[8,t]*m.ug[8,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3_t2[2]* m.inertia_constant[8]*m.genD_Pmax[8]*m.ug[8,t]) + bus2_seg3_t2[3] + (1 - m.v02_4_t2[2,t])*A >= m.t02_4_t2[2,t]
m.bus02_4_u3_t2 = Constraint(m.PERIOD,rule = bus02_4_u3_t2)

def bus02_4_t23_t2(m,t):
    return m.t02_4_t2[2,t] <= m.t02_4_t2[3,t]
m.bus02_4_t23_t2 = Constraint(m.PERIOD,rule = bus02_4_t23_t2)

def bus02_4_t32_t2(m,t):
    return m.t02_4_t2[3,t]<=m.t02_4_t2[2,t] + m.v02_4_t2[3,t]*A
m.bus02_4_t32_t2 = Constraint(m.PERIOD,rule = bus02_4_t32_t2)

def bus02_4_l4_t2(m,t):
    return bus2_seg4_t2[0]*100*m.Pg[8,t]*m.ug[8,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4_t2[2]* m.inertia_constant[8]*m.genD_Pmax[8]*m.ug[8,t]) + bus2_seg4_t2[3] <= m.t02_4_t2[3,t]
m.bus02_4_l4_t2 = Constraint(m.PERIOD,rule = bus02_4_l4_t2)

def bus02_4_u4_t2(m,t):
    return bus2_seg4_t2[0]*100*m.Pg[8,t]*m.ug[8,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4_t2[2]* m.inertia_constant[8]*m.genD_Pmax[8]*m.ug[8,t]) + bus2_seg4_t2[3] + (1 - m.v02_4_t2[3,t])*A >= m.t02_4_t2[3,t]
m.bus02_4_u4_t2 = Constraint(m.PERIOD,rule = bus02_4_u4_t2)

def bus02_5_R_t2(m,t):
    return m.t02_5_t2[3,t] <= RoCoF
m.bus02_5_R_t2 = Constraint(m.PERIOD, rule = bus02_5_R_t2)

def bus02_5_l1_t2(m,t):
    return bus2_seg1_t2[0]*100*m.Pg[9,t]*m.ug[9,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1_t2[2]* m.inertia_constant[9]*m.genD_Pmax[9]*m.ug[9,t]) + bus2_seg1_t2[3] <= m.t02_5_t2[1,t]
m.bus02_5_l1_t2 = Constraint(m.PERIOD, rule = bus02_5_l1_t2)

def bus02_5_u1_t2(m,t):
    return bus2_seg1_t2[0]*100*m.Pg[9,t]*m.ug[9,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1_t2[2]* m.inertia_constant[9]*m.genD_Pmax[9]*m.ug[9,t]) + bus2_seg1_t2[3] + m.v02_5_t2[1,t]*A >= m.t02_5_t2[1,t]
m.bus02_5_u1_t2 = Constraint(m.PERIOD, rule = bus02_5_u1_t2)

def bus02_5_l2_t2(m,t):
    return bus2_seg2_t2[0]*100*m.Pg[9,t]*m.ug[9,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2_t2[2]* m.inertia_constant[9]*m.genD_Pmax[9]*m.ug[9,t]) + bus2_seg2_t2[3] <= m.t02_5_t2[1,t]
m.bus02_5_l2_t2 = Constraint(m.PERIOD,rule = bus02_5_l2_t2)

def bus02_5_u2_t2(m,t):
    return bus2_seg2_t2[0]*100*m.Pg[9,t]*m.ug[9,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2_t2[2]* m.inertia_constant[9]*m.genD_Pmax[9]*m.ug[9,t]) + bus2_seg2_t2[3] + (1 - m.v02_5_t2[1,t])*A >= m.t02_5_t2[1,t]
m.bus02_5_u2_t2 = Constraint(m.PERIOD,rule = bus02_5_u2_t2)

def bus02_5_t12_t2(m,t):
    return m.t02_5_t2[1,t] <= m.t02_5_t2[2,t]
m.bus02_5_t12_t2 = Constraint(m.PERIOD,rule = bus02_5_t12_t2)

def bus02_5_t21_t2(m,t):
    return m.t02_5_t2[2,t]<=m.t02_5_t2[1,t] + m.v02_5_t2[2,t]*A
m.bus02_5_t21_t2 = Constraint(m.PERIOD,rule = bus02_5_t21_t2)

def bus02_5_l3_t2(m,t):
    return bus2_seg3_t2[0]*100*m.Pg[9,t]*m.ug[9,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3_t2[2]* m.inertia_constant[9]*m.genD_Pmax[9]*m.ug[9,t]) + bus2_seg3_t2[3]  <= m.t02_5_t2[2,t]
m.bus02_5_l3_t2 = Constraint(m.PERIOD,rule = bus02_5_l3_t2)

def bus02_5_u3_t2(m,t):
    return bus2_seg3_t2[0]*100*m.Pg[9,t]*m.ug[9,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3_t2[2]* m.inertia_constant[9]*m.genD_Pmax[9]*m.ug[9,t]) + bus2_seg3_t2[3] + (1 - m.v02_5_t2[2,t])*A >= m.t02_5_t2[2,t]
m.bus02_5_u3_t2 = Constraint(m.PERIOD,rule = bus02_5_u3_t2)

def bus02_5_t23_t2(m,t):
    return m.t02_5_t2[2,t] <= m.t02_5_t2[3,t]
m.bus02_5_t23_t2 = Constraint(m.PERIOD,rule = bus02_5_t23_t2)

def bus02_5_t32_t2(m,t):
    return m.t02_5_t2[3,t]<=m.t02_5_t2[2,t] + m.v02_5_t2[3,t]*A
m.bus02_5_t32_t2 = Constraint(m.PERIOD,rule = bus02_5_t32_t2)

def bus02_5_l4_t2(m,t):
    return bus2_seg4_t2[0]*100*m.Pg[9,t]*m.ug[9,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4_t2[2]* m.inertia_constant[9]*m.genD_Pmax[9]*m.ug[9,t]) + bus2_seg4_t2[3] <= m.t02_5_t2[3,t]
m.bus02_5_l4_t2 = Constraint(m.PERIOD,rule = bus02_5_l4_t2)

def bus02_5_u4_t2(m,t):
    return bus2_seg4_t2[0]*100*m.Pg[9,t]*m.ug[9,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4_t2[2]* m.inertia_constant[9]*m.genD_Pmax[9]*m.ug[9,t]) + bus2_seg4_t2[3] + (1 - m.v02_5_t2[3,t])*A >= m.t02_5_t2[3,t]
m.bus02_5_u4_t2 = Constraint(m.PERIOD,rule = bus02_5_u4_t2)

def bus02_6_R_t2(m,t):
    return m.t02_6_t2[3,t] <= RoCoF
m.bus02_6_R_t2 = Constraint(m.PERIOD, rule = bus02_6_R_t2)

def bus02_6_l1_t2(m,t):
    return bus2_seg1_t2[0]*100*m.Pg[10,t]*m.ug[10,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1_t2[2]* m.inertia_constant[10]*m.genD_Pmax[10]*m.ug[10,t]) + bus2_seg1_t2[3] <= m.t02_6_t2[1,t]
m.bus02_6_l1_t2 = Constraint(m.PERIOD, rule = bus02_6_l1_t2)

def bus02_6_u1_t2(m,t):
    return bus2_seg1_t2[0]*100*m.Pg[10,t]*m.ug[10,t] + bus2_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg1_t2[2]* m.inertia_constant[10]*m.genD_Pmax[10]*m.ug[10,t]) + bus2_seg1_t2[3] + m.v02_6_t2[1,t]*A >= m.t02_6_t2[1,t]
m.bus02_6_u1_t2 = Constraint(m.PERIOD, rule = bus02_6_u1_t2)

def bus02_6_l2_t2(m,t):
    return bus2_seg2_t2[0]*100*m.Pg[10,t]*m.ug[10,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2_t2[2]* m.inertia_constant[10]*m.genD_Pmax[10]*m.ug[10,t]) + bus2_seg2_t2[3] <= m.t02_6_t2[1,t]
m.bus02_6_l2_t2 = Constraint(m.PERIOD,rule = bus02_6_l2_t2)

def bus02_6_u2_t2(m,t):
    return bus2_seg2_t2[0]*100*m.Pg[10,t]*m.ug[10,t] + bus2_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg2_t2[2]* m.inertia_constant[10]*m.genD_Pmax[10]*m.ug[10,t]) + bus2_seg2_t2[3] + (1 - m.v02_6_t2[1,t])*A >= m.t02_6_t2[1,t]
m.bus02_6_u2_t2 = Constraint(m.PERIOD,rule = bus02_6_u2_t2)

def bus02_6_t12_t2(m,t):
    return m.t02_6_t2[1,t] <= m.t02_6_t2[2,t]
m.bus02_6_t12_t2 = Constraint(m.PERIOD,rule = bus02_6_t12_t2)

def bus02_6_t21_t2(m,t):
    return m.t02_6_t2[2,t]<=m.t02_6_t2[1,t] + m.v02_6_t2[2,t]*A
m.bus02_6_t21_t2 = Constraint(m.PERIOD,rule = bus02_6_t21_t2)

def bus02_6_l3_t2(m,t):
    return bus2_seg3_t2[0]*100*m.Pg[10,t]*m.ug[10,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3_t2[2]* m.inertia_constant[10]*m.genD_Pmax[10]*m.ug[10,t]) + bus2_seg3_t2[3]  <= m.t02_6_t2[2,t]
m.bus02_6_l3_t2 = Constraint(m.PERIOD,rule = bus02_6_l3_t2)

def bus02_6_u3_t2(m,t):
    return bus2_seg3_t2[0]*100*m.Pg[10,t]*m.ug[10,t] + bus2_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg3_t2[2]* m.inertia_constant[10]*m.genD_Pmax[10]*m.ug[10,t]) + bus2_seg3_t2[3] + (1 - m.v02_6_t2[2,t])*A >= m.t02_6_t2[2,t]
m.bus02_6_u3_t2 = Constraint(m.PERIOD,rule = bus02_6_u3_t2)

def bus02_6_t23_t2(m,t):
    return m.t02_6_t2[2,t] <= m.t02_6_t2[3,t]
m.bus02_6_t23_t2 = Constraint(m.PERIOD,rule = bus02_6_t23_t2)

def bus02_6_t32_t2(m,t):
    return m.t02_6_t2[3,t]<=m.t02_6_t2[2,t] + m.v02_6_t2[3,t]*A
m.bus02_6_t32_t2 = Constraint(m.PERIOD,rule = bus02_6_t32_t2)

def bus02_6_l4_t2(m,t):
    return bus2_seg4_t2[0]*100*m.Pg[10,t]*m.ug[10,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4_t2[2]* m.inertia_constant[10]*m.genD_Pmax[10]*m.ug[10,t]) + bus2_seg4_t2[3] <= m.t02_6_t2[3,t]
m.bus02_6_l4_t2 = Constraint(m.PERIOD,rule = bus02_6_l4_t2)

def bus02_6_u4_t2(m,t):
    return bus2_seg4_t2[0]*100*m.Pg[10,t]*m.ug[10,t] + bus2_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus2_seg4_t2[2]* m.inertia_constant[10]*m.genD_Pmax[10]*m.ug[10,t]) + bus2_seg4_t2[3] + (1 - m.v02_6_t2[3,t])*A >= m.t02_6_t2[3,t]
m.bus02_6_u4_t2 = Constraint(m.PERIOD,rule = bus02_6_u4_t2)

# Node 07 locational RoCoF constraints

def bus07_1_R_t2(m,t):
    return m.t07_1_t2[3,t] <= RoCoF
m.bus07_1_R_t2 = Constraint(m.PERIOD, rule = bus07_1_R_t2)

def bus07_1_l1_t2(m,t):
    return bus7_seg1_t2[0]*100*m.Pg[11,t]*m.ug[11,t] + bus7_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg1_t2[2]* m.inertia_constant[11]*m.genD_Pmax[11]*m.ug[11,t]) + bus7_seg1_t2[3]<= m.t07_1_t2[1,t]
m.bus07_1_l1_t2 = Constraint(m.PERIOD, rule = bus07_1_l1_t2)

def bus07_1_u1_t2(m,t):
    return bus7_seg1_t2[0]*100*m.Pg[11,t]*m.ug[11,t] + bus7_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg1_t2[2]* m.inertia_constant[11]*m.genD_Pmax[11]*m.ug[11,t]) + bus7_seg1_t2[3] + m.v07_1_t2[1,t]*A >= m.t07_1_t2[1,t]
m.bus07_1_u1_t2 = Constraint(m.PERIOD, rule = bus07_1_u1_t2)

def bus07_1_l2_t2(m,t):
    return bus7_seg2_t2[0]*100*m.Pg[11,t]*m.ug[11,t] + bus7_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg2_t2[2]* m.inertia_constant[11]*m.genD_Pmax[11]*m.ug[11,t]) + bus7_seg2_t2[3] <= m.t07_1_t2[1,t]
m.bus07_1_l2_t2 = Constraint(m.PERIOD,rule = bus07_1_l2_t2)

def bus07_1_u2_t2(m,t):
    return bus7_seg2_t2[0]*100*m.Pg[11,t]*m.ug[11,t] + bus7_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg2_t2[2]* m.inertia_constant[11]*m.genD_Pmax[11]*m.ug[11,t]) + bus7_seg2_t2[3] + (1 - m.v07_1_t2[1,t])*A >= m.t07_1_t2[1,t]
m.bus07_1_u2_t2 = Constraint(m.PERIOD,rule = bus07_1_u2_t2)

def bus07_1_t12_t2(m,t):
    return m.t07_1_t2[1,t] <= m.t07_1_t2[2,t]
m.bus07_1_t12_t2 = Constraint(m.PERIOD,rule = bus07_1_t12_t2)

def bus07_1_t21_t2(m,t):
    return m.t07_1_t2[2,t]<=m.t07_1_t2[1,t] + m.v07_1_t2[2,t]*A
m.bus07_1_t21_t2 = Constraint(m.PERIOD,rule = bus07_1_t21_t2)

def bus07_1_l3_t2(m,t):
    return bus7_seg3_t2[0]*100*m.Pg[11,t]*m.ug[11,t] + bus7_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg3_t2[2]* m.inertia_constant[11]*m.genD_Pmax[11]*m.ug[11,t]) + bus7_seg3_t2[3] <= m.t07_1_t2[2,t]
m.bus07_1_l3_t2 = Constraint(m.PERIOD,rule = bus07_1_l3_t2)

def bus07_1_u3_t2(m,t):
    return bus7_seg3_t2[0]*100*m.Pg[11,t]*m.ug[11,t] + bus7_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg3_t2[2]* m.inertia_constant[11]*m.genD_Pmax[11]*m.ug[11,t]) + bus7_seg3_t2[3]+ (1 - m.v07_1_t2[2,t])*A >= m.t07_1_t2[2,t]
m.bus07_1_u3_t2 = Constraint(m.PERIOD,rule = bus07_1_u3_t2)

def bus07_1_t23_t2(m,t):
    return m.t07_1_t2[2,t] <= m.t07_1_t2[3,t]
m.bus07_1_t23_t2 = Constraint(m.PERIOD,rule = bus07_1_t23_t2)

def bus07_1_t32_t2(m,t):
    return m.t07_1_t2[3,t]<=m.t07_1_t2[2,t] + m.v07_1_t2[3,t]*A
m.bus07_1_t32_t2 = Constraint(m.PERIOD,rule = bus07_1_t32_t2)

def bus07_1_l4_t2(m,t):
    return bus7_seg4_t2[0]*100*m.Pg[11,t]*m.ug[11,t] + bus7_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg4_t2[2]* m.inertia_constant[11]*m.genD_Pmax[11]*m.ug[11,t]) + bus7_seg4_t2[3] <= m.t07_1_t2[3,t]
m.bus07_1_l4_t2 = Constraint(m.PERIOD,rule = bus07_1_l4_t2)

def bus07_1_u4_t2(m,t):
    return bus7_seg4_t2[0]*100*m.Pg[11,t]*m.ug[11,t] + bus7_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg4_t2[2]* m.inertia_constant[11]*m.genD_Pmax[11]*m.ug[11,t]) + bus7_seg4_t2[3] + (1 - m.v07_1_t2[3,t])*A >= m.t07_1_t2[3,t]
m.bus07_1_u4_t2 = Constraint(m.PERIOD,rule = bus07_1_u4_t2)

def bus07_2_R_t2(m,t):
    return m.t07_2_t2[3,t] <= RoCoF
m.bus07_2_R_t2 = Constraint(m.PERIOD, rule = bus07_2_R_t2)

def bus07_2_l1_t2(m,t):
    return bus7_seg1_t2[0]*100*m.Pg[12,t]*m.ug[12,t] + bus7_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg1_t2[2]* m.inertia_constant[12]*m.genD_Pmax[12]*m.ug[12,t]) + bus7_seg1_t2[3] <= m.t07_2_t2[1,t]
m.bus07_2_l1_t2 = Constraint(m.PERIOD, rule = bus07_2_l1_t2)

def bus07_2_u1_t2(m,t):
    return bus7_seg1_t2[0]*100*m.Pg[12,t]*m.ug[12,t] + bus7_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg1_t2[2]* m.inertia_constant[12]*m.genD_Pmax[12]*m.ug[12,t]) + bus7_seg1_t2[3]+ m.v07_2_t2[1,t]*A >= m.t07_2_t2[1,t]
m.bus07_2_u1_t2 = Constraint(m.PERIOD, rule = bus07_2_u1_t2)

def bus07_2_l2_t2(m,t):
    return bus7_seg2_t2[0]*100*m.Pg[12,t]*m.ug[12,t] + bus7_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg2_t2[2]* m.inertia_constant[12]*m.genD_Pmax[12]*m.ug[12,t]) + bus7_seg2_t2[3] <= m.t07_2_t2[1,t]
m.bus07_2_l2_t2 = Constraint(m.PERIOD,rule = bus07_2_l2_t2)

def bus07_2_u2_t2(m,t):
    return bus7_seg2_t2[0]*100*m.Pg[12,t]*m.ug[12,t] + bus7_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg2_t2[2]* m.inertia_constant[12]*m.genD_Pmax[12]*m.ug[12,t]) + bus7_seg2_t2[3] + (1 - m.v07_2_t2[1,t])*A >= m.t07_2_t2[1,t]
m.bus07_2_u2_t2 = Constraint(m.PERIOD,rule = bus07_2_u2_t2)

def bus07_2_t12_t2(m,t):
    return m.t07_2_t2[1,t] <= m.t07_2_t2[2,t]
m.bus07_2_t12_t2 = Constraint(m.PERIOD,rule = bus07_2_t12_t2)

def bus07_2_t21_t2(m,t):
    return m.t07_2_t2[2,t]<=m.t07_2_t2[1,t] + m.v07_2_t2[2,t]*A
m.bus07_2_t21_t2 = Constraint(m.PERIOD,rule = bus07_2_t21_t2)

def bus07_2_l3_t2(m,t):
    return bus7_seg3_t2[0]*100*m.Pg[12,t]*m.ug[12,t] + bus7_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg3_t2[2]* m.inertia_constant[12]*m.genD_Pmax[12]*m.ug[12,t]) + bus7_seg3_t2[3] <= m.t07_2_t2[2,t]
m.bus07_2_l3_t2 = Constraint(m.PERIOD,rule = bus07_2_l3_t2)

def bus07_2_u3_t2(m,t):
    return bus7_seg3_t2[0]*100*m.Pg[12,t]*m.ug[12,t] + bus7_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg3_t2[2]* m.inertia_constant[12]*m.genD_Pmax[12]*m.ug[12,t]) + bus7_seg3_t2[3] + (1 - m.v07_2_t2[2,t])*A >= m.t07_2_t2[2,t]
m.bus07_2_u3_t2 = Constraint(m.PERIOD,rule = bus07_2_u3_t2)

def bus07_2_t23_t2(m,t):
    return m.t07_2_t2[2,t] <= m.t07_2_t2[3,t]
m.bus07_2_t23_t2 = Constraint(m.PERIOD,rule = bus07_2_t23_t2)

def bus07_2_t32_t2(m,t):
    return m.t07_2_t2[3,t]<=m.t07_2_t2[2,t] + m.v07_2_t2[3,t]*A
m.bus07_2_t32_t2 = Constraint(m.PERIOD,rule = bus07_2_t32_t2)

def bus07_2_l4_t2(m,t):
    return bus7_seg4_t2[0]*100*m.Pg[12,t]*m.ug[12,t] + bus7_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg4_t2[2]* m.inertia_constant[12]*m.genD_Pmax[12]*m.ug[12,t]) + bus7_seg4_t2[3] <= m.t07_2_t2[3,t]
m.bus07_2_l4_t2 = Constraint(m.PERIOD,rule = bus07_2_l4_t2)

def bus07_2_u4_t2(m,t):
    return bus7_seg4_t2[0]*100*m.Pg[12,t]*m.ug[12,t] + bus7_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg4_t2[2]* m.inertia_constant[12]*m.genD_Pmax[12]*m.ug[12,t]) + bus7_seg4_t2[3] + (1 - m.v07_2_t2[3,t])*A >= m.t07_2_t2[3,t]
m.bus07_2_u4_t2 = Constraint(m.PERIOD,rule = bus07_2_u4_t2)

def bus07_3_R_t2(m,t):
    return m.t07_3_t2[3,t] <= RoCoF
m.bus07_3_R_t2 = Constraint(m.PERIOD, rule = bus07_3_R_t2)

def bus07_3_l1_t2(m,t):
    return bus7_seg1_t2[0]*100*m.Pg[13,t]*m.ug[13,t] + bus7_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg1_t2[2]* m.inertia_constant[13]*m.genD_Pmax[13]*m.ug[13,t]) + bus7_seg1_t2[3] <= m.t07_3_t2[1,t]
m.bus07_3_l1_t2 = Constraint(m.PERIOD, rule = bus07_3_l1_t2)

def bus07_3_u1_t2(m,t):
    return bus7_seg1_t2[0]*100*m.Pg[13,t]*m.ug[13,t] + bus7_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg1_t2[2]* m.inertia_constant[13]*m.genD_Pmax[13]*m.ug[13,t]) + bus7_seg1_t2[3] + m.v07_3_t2[1,t]*A >= m.t07_3_t2[1,t]
m.bus07_3_u1_t2 = Constraint(m.PERIOD, rule = bus07_3_u1_t2)

def bus07_3_l2_t2(m,t):
    return bus7_seg2_t2[0]*100*m.Pg[13,t]*m.ug[13,t] + bus7_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg2_t2[2]* m.inertia_constant[13]*m.genD_Pmax[13]*m.ug[13,t]) + bus7_seg2_t2[3] <= m.t07_3_t2[1,t]
m.bus07_3_l2_t2 = Constraint(m.PERIOD,rule = bus07_3_l2_t2)

def bus07_3_u2_t2(m,t):
    return bus7_seg2_t2[0]*100*m.Pg[13,t]*m.ug[13,t] + bus7_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg2_t2[2]* m.inertia_constant[13]*m.genD_Pmax[13]*m.ug[13,t]) + bus7_seg2_t2[3] + (1 - m.v07_3_t2[1,t])*A >= m.t07_3_t2[1,t]
m.bus07_3_u2_t2 = Constraint(m.PERIOD,rule = bus07_3_u2_t2)

def bus07_3_t12_t2(m,t):
    return m.t07_3_t2[1,t] <= m.t07_3_t2[2,t]
m.bus07_3_t12_t2 = Constraint(m.PERIOD,rule = bus07_3_t12_t2)

def bus07_3_t21_t2(m,t):
    return m.t07_3_t2[2,t]<=m.t07_3_t2[1,t] + m.v07_3_t2[2,t]*A
m.bus07_3_t21_t2 = Constraint(m.PERIOD,rule = bus07_3_t21_t2)

def bus07_3_l3_t2(m,t):
    return bus7_seg3_t2[0]*100*m.Pg[13,t]*m.ug[13,t] + bus7_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg3_t2[2]* m.inertia_constant[13]*m.genD_Pmax[13]*m.ug[13,t]) + bus7_seg3_t2[3]  <= m.t07_3_t2[2,t]
m.bus07_3_l3_t2 = Constraint(m.PERIOD,rule = bus07_3_l3_t2)

def bus07_3_u3_t2(m,t):
    return bus7_seg3_t2[0]*100*m.Pg[13,t]*m.ug[13,t] + bus7_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg3_t2[2]* m.inertia_constant[13]*m.genD_Pmax[13]*m.ug[13,t]) + bus7_seg3_t2[3]  + (1 - m.v07_3_t2[2,t])*A >= m.t07_3_t2[2,t]
m.bus07_3_u3_t2 = Constraint(m.PERIOD,rule = bus07_3_u3_t2)

def bus07_3_t23_t2(m,t):
    return m.t07_3_t2[2,t] <= m.t07_3_t2[3,t]
m.bus07_3_t23_t2 = Constraint(m.PERIOD,rule = bus07_3_t23_t2)

def bus07_3_t32_t2(m,t):
    return m.t07_3_t2[3,t]<=m.t07_3_t2[2,t] + m.v07_3_t2[3,t]*A
m.bus07_3_t32_t2 = Constraint(m.PERIOD,rule = bus07_3_t32_t2)

def bus07_3_l4_t2(m,t):
    return bus7_seg4_t2[0]*100*m.Pg[13,t]*m.ug[13,t] + bus7_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg4_t2[2]* m.inertia_constant[13]*m.genD_Pmax[13]*m.ug[13,t]) + bus7_seg4_t2[3] <= m.t07_3_t2[3,t]
m.bus07_3_l4_t2 = Constraint(m.PERIOD,rule = bus07_3_l4_t2)

def bus07_3_u4_t2(m,t):
    return bus7_seg4_t2[0]*100*m.Pg[13,t]*m.ug[13,t] + bus7_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus7_seg4_t2[2]* m.inertia_constant[13]*m.genD_Pmax[13]*m.ug[13,t]) + bus7_seg4_t2[3] + (1 - m.v07_3_t2[3,t])*A >= m.t07_3_t2[3,t]
m.bus07_3_u4_t2 = Constraint(m.PERIOD,rule = bus07_3_u4_t2)

# Node 13 locational RoCoF constraints

def bus13_1_R_t2(m,t):
    return m.t13_1_t2[3,t] <= RoCoF
m.bus13_1_R_t2 = Constraint(m.PERIOD, rule = bus13_1_R_t2)

def bus13_1_l1_t2(m,t):
    return bus13_seg1_t2[0]*100*m.Pg[14,t]*m.ug[14,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1_t2[2]* m.inertia_constant[14]*m.genD_Pmax[14]*m.ug[14,t]) + bus13_seg1_t2[3] <= m.t13_1_t2[1,t]
m.bus13_1_l1_t2 = Constraint(m.PERIOD, rule = bus13_1_l1_t2)

def bus13_1_u1_t2(m,t):
    return bus13_seg1_t2[0]*100*m.Pg[14,t]*m.ug[14,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1_t2[2]* m.inertia_constant[14]*m.genD_Pmax[14]*m.ug[14,t]) + bus13_seg1_t2[3] + m.v13_1_t2[1,t]*A >= m.t13_1_t2[1,t]
m.bus13_1_u1_t2 = Constraint(m.PERIOD, rule = bus13_1_u1_t2)

def bus13_1_l2_t2(m,t):
    return bus13_seg2_t2[0]*100*m.Pg[14,t]*m.ug[14,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2_t2[2]* m.inertia_constant[14]*m.genD_Pmax[14]*m.ug[14,t]) + bus13_seg2_t2[3] <= m.t13_1_t2[1,t]
m.bus13_1_l2_t2 = Constraint(m.PERIOD,rule = bus13_1_l2_t2)

def bus13_1_u2_t2(m,t):
    return bus13_seg2_t2[0]*100*m.Pg[14,t]*m.ug[14,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2_t2[2]* m.inertia_constant[14]*m.genD_Pmax[14]*m.ug[14,t]) + bus13_seg2_t2[3] + (1 - m.v13_1_t2[1,t])*A >= m.t13_1_t2[1,t]
m.bus13_1_u2_t2 = Constraint(m.PERIOD,rule = bus13_1_u2_t2)

def bus13_1_t12_t2(m,t):
    return m.t13_1_t2[1,t] <= m.t13_1_t2[2,t]
m.bus13_1_t12_t2 = Constraint(m.PERIOD,rule = bus13_1_t12_t2)

def bus13_1_t21_t2(m,t):
    return m.t13_1_t2[2,t]<=m.t13_1_t2[1,t] + m.v13_1_t2[2,t]*A
m.bus13_1_t21_t2 = Constraint(m.PERIOD,rule = bus13_1_t21_t2)

def bus13_1_l3_t2(m,t):
    return bus13_seg3_t2[0]*100*m.Pg[14,t]*m.ug[14,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3_t2[2]* m.inertia_constant[14]*m.genD_Pmax[14]*m.ug[14,t]) + bus13_seg3_t2[3] <= m.t13_1_t2[2,t]
m.bus13_1_l3_t2 = Constraint(m.PERIOD,rule = bus13_1_l3_t2)

def bus13_1_u3_t2(m,t):
    return bus13_seg3_t2[0]*100*m.Pg[14,t]*m.ug[14,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3_t2[2]* m.inertia_constant[14]*m.genD_Pmax[14]*m.ug[14,t]) + bus13_seg3_t2[3] + (1 - m.v13_1_t2[2,t])*A >= m.t13_1_t2[2,t]
m.bus13_1_u3_t2 = Constraint(m.PERIOD,rule = bus13_1_u3_t2)

def bus13_1_t23_t2(m,t):
    return m.t13_1_t2[2,t] <= m.t13_1_t2[3,t]
m.bus13_1_t23_t2 = Constraint(m.PERIOD,rule = bus13_1_t23_t2)

def bus13_1_t32_t2(m,t):
    return m.t13_1_t2[3,t]<=m.t13_1_t2[2,t] + m.v13_1_t2[3,t]*A
m.bus13_1_t32_t2 = Constraint(m.PERIOD,rule = bus13_1_t32_t2)

def bus13_1_l4_t2(m,t):
    return bus13_seg4_t2[0]*100*m.Pg[14,t]*m.ug[14,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4_t2[2]* m.inertia_constant[14]*m.genD_Pmax[14]*m.ug[14,t]) + bus13_seg4_t2[3] <= m.t13_1_t2[3,t]
m.bus13_1_l4_t2 = Constraint(m.PERIOD,rule = bus13_1_l4_t2)

def bus13_1_u4_t2(m,t):
    return bus13_seg4_t2[0]*100*m.Pg[14,t]*m.ug[14,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4_t2[2]* m.inertia_constant[14]*m.genD_Pmax[14]*m.ug[14,t]) + bus13_seg4_t2[3] + (1 - m.v13_1_t2[3,t])*A >= m.t13_1_t2[3,t]
m.bus13_1_u4_t2 = Constraint(m.PERIOD,rule = bus13_1_u4_t2)

def bus13_2_R_t2(m,t):
    return m.t13_2_t2[3,t] <= RoCoF
m.bus13_2_R_t2 = Constraint(m.PERIOD, rule = bus13_2_R_t2)

def bus13_2_l1_t2(m,t):
    return bus13_seg1_t2[0]*100*m.Pg[15,t]*m.ug[15,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1_t2[2]* m.inertia_constant[15]*m.genD_Pmax[15]*m.ug[15,t]) + bus13_seg1_t2[3] <= m.t13_2_t2[1,t]
m.bus13_2_l1_t2 = Constraint(m.PERIOD, rule = bus13_2_l1_t2)

def bus13_2_u1_t2(m,t):
    return bus13_seg1_t2[0]*100*m.Pg[15,t]*m.ug[15,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1_t2[2]* m.inertia_constant[15]*m.genD_Pmax[15]*m.ug[15,t]) + bus13_seg1_t2[3] + m.v13_2_t2[1,t]*A >= m.t13_2_t2[1,t]
m.bus13_2_u1_t2 = Constraint(m.PERIOD, rule = bus13_2_u1_t2)

def bus13_2_l2_t2(m,t):
    return bus13_seg2_t2[0]*100*m.Pg[15,t]*m.ug[15,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2_t2[2]* m.inertia_constant[15]*m.genD_Pmax[15]*m.ug[15,t]) + bus13_seg2_t2[3] <= m.t13_2_t2[1,t]
m.bus13_2_l2_t2 = Constraint(m.PERIOD,rule = bus13_2_l2_t2)

def bus13_2_u2_t2(m,t):
    return bus13_seg2_t2[0]*100*m.Pg[15,t]*m.ug[15,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2_t2[2]* m.inertia_constant[15]*m.genD_Pmax[15]*m.ug[15,t]) + bus13_seg2_t2[3]+ (1 - m.v13_2_t2[1,t])*A >= m.t13_2_t2[1,t]
m.bus13_2_u2_t2 = Constraint(m.PERIOD,rule = bus13_2_u2_t2)

def bus13_2_t12_t2(m,t):
    return m.t13_2_t2[1,t] <= m.t13_2_t2[2,t]
m.bus13_2_t12_t2 = Constraint(m.PERIOD,rule = bus13_2_t12_t2)

def bus13_2_t21_t2(m,t):
    return m.t13_2_t2[2,t]<=m.t13_2_t2[1,t] + m.v13_2_t2[2,t]*A
m.bus13_2_t21_t2 = Constraint(m.PERIOD,rule = bus13_2_t21_t2)

def bus13_2_l3_t2(m,t):
    return bus13_seg3_t2[0]*100*m.Pg[15,t]*m.ug[15,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3_t2[2]* m.inertia_constant[15]*m.genD_Pmax[15]*m.ug[15,t]) + bus13_seg3_t2[3]  <= m.t13_2_t2[2,t]
m.bus13_2_l3_t2 = Constraint(m.PERIOD,rule = bus13_2_l3_t2)

def bus13_2_u3_t2(m,t):
    return bus13_seg3_t2[0]*100*m.Pg[15,t]*m.ug[15,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3_t2[2]* m.inertia_constant[15]*m.genD_Pmax[15]*m.ug[15,t]) + bus13_seg3_t2[3] + (1 - m.v13_2_t2[2,t])*A >= m.t13_2_t2[2,t]
m.bus13_2_u3_t2 = Constraint(m.PERIOD,rule = bus13_2_u3_t2)

def bus13_2_t23_t2(m,t):
    return m.t13_2_t2[2,t] <= m.t13_2_t2[3,t]
m.bus13_2_t23_t2 = Constraint(m.PERIOD,rule = bus13_2_t23_t2)

def bus13_2_t32_t2(m,t):
    return m.t13_2_t2[3,t]<=m.t13_2_t2[2,t] + m.v13_2_t2[3,t]*A
m.bus13_2_t32_t2 = Constraint(m.PERIOD,rule = bus13_2_t32_t2)

def bus13_2_l4_t2(m,t):
    return bus13_seg4_t2[0]*100*m.Pg[15,t]*m.ug[15,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4_t2[2]* m.inertia_constant[15]*m.genD_Pmax[15]*m.ug[15,t]) + bus13_seg4_t2[3] <= m.t13_2_t2[3,t]
m.bus13_2_l4_t2 = Constraint(m.PERIOD,rule = bus13_2_l4_t2)

def bus13_2_u4_t2(m,t):
    return bus13_seg4_t2[0]*100*m.Pg[15,t]*m.ug[15,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4_t2[2]* m.inertia_constant[15]*m.genD_Pmax[15]*m.ug[15,t]) + bus13_seg4_t2[3] + (1 - m.v13_2_t2[3,t])*A >= m.t13_2_t2[3,t]
m.bus13_2_u4_t2 = Constraint(m.PERIOD,rule = bus13_2_u4_t2)

def bus13_3_R_t2(m,t):
    return m.t13_3_t2[3,t] <= RoCoF
m.bus13_3_R_t2 = Constraint(m.PERIOD, rule = bus13_3_R_t2)

def bus13_3_l1_t2(m,t):
    return bus13_seg1_t2[0]*100*m.Pg[16,t]*m.ug[16,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1_t2[2]* m.inertia_constant[16]*m.genD_Pmax[16]*m.ug[16,t]) + bus13_seg1_t2[3] <= m.t13_3_t2[1,t]
m.bus13_3_l1_t2 = Constraint(m.PERIOD, rule = bus13_3_l1_t2)

def bus13_3_u1_t2(m,t):
    return bus13_seg1_t2[0]*100*m.Pg[16,t]*m.ug[16,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1_t2[2]* m.inertia_constant[16]*m.genD_Pmax[16]*m.ug[16,t]) + bus13_seg1_t2[3] + m.v13_3_t2[1,t]*A >= m.t13_3_t2[1,t]
m.bus13_3_u1_t2 = Constraint(m.PERIOD, rule = bus13_3_u1_t2)

def bus13_3_l2_t2(m,t):
    return bus13_seg2_t2[0]*100*m.Pg[16,t]*m.ug[16,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2_t2[2]* m.inertia_constant[16]*m.genD_Pmax[16]*m.ug[16,t]) + bus13_seg2_t2[3] <= m.t13_3_t2[1,t]
m.bus13_3_l2_t2 = Constraint(m.PERIOD,rule = bus13_3_l2_t2)

def bus13_3_u2_t2(m,t):
    return bus13_seg2_t2[0]*100*m.Pg[16,t]*m.ug[16,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2_t2[2]* m.inertia_constant[16]*m.genD_Pmax[16]*m.ug[16,t]) + bus13_seg2_t2[3] + (1 - m.v13_3_t2[1,t])*A >= m.t13_3_t2[1,t]
m.bus13_3_u2_t2 = Constraint(m.PERIOD,rule = bus13_3_u2_t2)

def bus13_3_t12_t2(m,t):
    return m.t13_3_t2[1,t] <= m.t13_3_t2[2,t]
m.bus13_3_t12_t2 = Constraint(m.PERIOD,rule = bus13_3_t12_t2)

def bus13_3_t21_t2(m,t):
    return m.t13_3_t2[2,t]<=m.t13_3_t2[1,t] + m.v13_3_t2[2,t]*A
m.bus13_3_t21_t2 = Constraint(m.PERIOD,rule = bus13_3_t21_t2)

def bus13_3_l3_t2(m,t):
    return bus13_seg3_t2[0]*100*m.Pg[16,t]*m.ug[16,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3_t2[2]* m.inertia_constant[16]*m.genD_Pmax[16]*m.ug[16,t]) + bus13_seg3_t2[3] <= m.t13_3_t2[2,t]
m.bus13_3_l3_t2 = Constraint(m.PERIOD,rule = bus13_3_l3_t2)

def bus13_3_u3_t2(m,t):
    return bus13_seg3_t2[0]*100*m.Pg[16,t]*m.ug[16,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3_t2[2]* m.inertia_constant[16]*m.genD_Pmax[16]*m.ug[16,t]) + bus13_seg3_t2[3] + (1 - m.v13_3_t2[2,t])*A >= m.t13_3_t2[2,t]
m.bus13_3_u3_t2 = Constraint(m.PERIOD,rule = bus13_3_u3_t2)

def bus13_3_t23_t2(m,t):
    return m.t13_3_t2[2,t] <= m.t13_3_t2[3,t]
m.bus13_3_t23_t2 = Constraint(m.PERIOD,rule = bus13_3_t23_t2)

def bus13_3_t32_t2(m,t):
    return m.t13_3_t2[3,t]<=m.t13_3_t2[2,t] + m.v13_3_t2[3,t]*A
m.bus13_3_t32_t2 = Constraint(m.PERIOD,rule = bus13_3_t32_t2)

def bus13_3_l4_t2(m,t):
    return bus13_seg4_t2[0]*100*m.Pg[16,t]*m.ug[16,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4_t2[2]* m.inertia_constant[16]*m.genD_Pmax[16]*m.ug[16,t]) + bus13_seg4_t2[3] <= m.t13_3_t2[3,t]
m.bus13_3_l4_t2 = Constraint(m.PERIOD,rule = bus13_3_l4_t2)

def bus13_3_u4_t2(m,t):
    return bus13_seg4_t2[0]*100*m.Pg[16,t]*m.ug[16,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4_t2[2]* m.inertia_constant[16]*m.genD_Pmax[16]*m.ug[16,t]) + bus13_seg4_t2[3] + (1 - m.v13_3_t2[3,t])*A >= m.t13_3_t2[3,t]
m.bus13_3_u4_t2 = Constraint(m.PERIOD,rule = bus13_3_u4_t2)

def bus13_4_R_t2(m,t):
    return m.t13_4_t2[3,t] <= RoCoF
m.bus13_4_R_t2 = Constraint(m.PERIOD, rule = bus13_4_R_t2)

def bus13_4_l1_t2(m,t):
    return bus13_seg1_t2[0]*100*m.Pg[17,t]*m.ug[17,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1_t2[2]* m.inertia_constant[17]*m.genD_Pmax[17]*m.ug[17,t]) + bus13_seg1_t2[3] <= m.t13_4_t2[1,t]
m.bus13_4_l1_t2 = Constraint(m.PERIOD, rule = bus13_4_l1_t2)

def bus13_4_u1_t2(m,t):
    return bus13_seg1_t2[0]*100*m.Pg[17,t]*m.ug[17,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1_t2[2]* m.inertia_constant[17]*m.genD_Pmax[17]*m.ug[17,t]) + bus13_seg1_t2[3] + m.v13_4_t2[1,t]*A >= m.t13_4_t2[1,t]
m.bus13_4_u1_t2 = Constraint(m.PERIOD, rule = bus13_4_u1_t2)

def bus13_4_l2_t2(m,t):
    return bus13_seg2_t2[0]*100*m.Pg[17,t]*m.ug[17,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2_t2[2]* m.inertia_constant[17]*m.genD_Pmax[17]*m.ug[17,t]) + bus13_seg2_t2[3] <= m.t13_4_t2[1,t]
m.bus13_4_l2_t2 = Constraint(m.PERIOD,rule = bus13_4_l2_t2)

def bus13_4_u2_t2(m,t):
    return bus13_seg2_t2[0]*100*m.Pg[17,t]*m.ug[17,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2_t2[2]* m.inertia_constant[17]*m.genD_Pmax[17]*m.ug[17,t]) + bus13_seg2_t2[3]+ (1 - m.v13_4_t2[1,t])*A >= m.t13_4_t2[1,t]
m.bus13_4_u2_t2 = Constraint(m.PERIOD,rule = bus13_4_u2_t2)

def bus13_4_t12_t2(m,t):
    return m.t13_4_t2[1,t] <= m.t13_4_t2[2,t]
m.bus13_4_t12_t2 = Constraint(m.PERIOD,rule = bus13_4_t12_t2)

def bus13_4_t21_t2(m,t):
    return m.t13_4_t2[2,t]<=m.t13_4_t2[1,t] + m.v13_4_t2[2,t]*A
m.bus13_4_t21_t2 = Constraint(m.PERIOD,rule = bus13_4_t21_t2)

def bus13_4_l3_t2(m,t):
    return bus13_seg3_t2[0]*100*m.Pg[17,t]*m.ug[17,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3_t2[2]* m.inertia_constant[17]*m.genD_Pmax[17]*m.ug[17,t]) + bus13_seg3_t2[3]  <= m.t13_4_t2[2,t]
m.bus13_4_l3_t2 = Constraint(m.PERIOD,rule = bus13_4_l3_t2)

def bus13_4_u3_t2(m,t):
    return bus13_seg3_t2[0]*100*m.Pg[17,t]*m.ug[17,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3_t2[2]* m.inertia_constant[17]*m.genD_Pmax[17]*m.ug[17,t]) + bus13_seg3_t2[3] + (1 - m.v13_4_t2[2,t])*A >= m.t13_4_t2[2,t]
m.bus13_4_u3_t2 = Constraint(m.PERIOD,rule = bus13_4_u3_t2)

def bus13_4_t23_t2(m,t):
    return m.t13_4_t2[2,t] <= m.t13_4_t2[3,t]
m.bus13_4_t23_t2 = Constraint(m.PERIOD,rule = bus13_4_t23_t2)

def bus13_4_t32_t2(m,t):
    return m.t13_4_t2[3,t]<=m.t13_4_t2[2,t] + m.v13_4_t2[3,t]*A
m.bus13_4_t32_t2 = Constraint(m.PERIOD,rule = bus13_4_t32_t2)

def bus13_4_l4_t2(m,t):
    return bus13_seg4_t2[0]*100*m.Pg[17,t]*m.ug[17,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4_t2[2]* m.inertia_constant[17]*m.genD_Pmax[17]*m.ug[17,t]) + bus13_seg4_t2[3] <= m.t13_4_t2[3,t]
m.bus13_4_l4_t2 = Constraint(m.PERIOD,rule = bus13_4_l4_t2)

def bus13_4_u4_t2(m,t):
    return bus13_seg4_t2[0]*100*m.Pg[17,t]*m.ug[17,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4_t2[2]* m.inertia_constant[17]*m.genD_Pmax[17]*m.ug[17,t]) + bus13_seg4_t2[3] + (1 - m.v13_4_t2[3,t])*A >= m.t13_4_t2[3,t]
m.bus13_4_u4_t2 = Constraint(m.PERIOD,rule = bus13_4_u4_t2)

def bus13_5_R_t2(m,t):
    return m.t13_5_t2[3,t] <= RoCoF
m.bus13_5_R_t2 = Constraint(m.PERIOD, rule = bus13_5_R_t2)

def bus13_5_l1_t2(m,t):
    return bus13_seg1_t2[0]*100*m.Pg[18,t]*m.ug[18,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1_t2[2]* m.inertia_constant[18]*m.genD_Pmax[18]*m.ug[18,t]) + bus13_seg1_t2[3] <= m.t13_5_t2[1,t]
m.bus13_5_l1_t2 = Constraint(m.PERIOD, rule = bus13_5_l1_t2)

def bus13_5_u1_t2(m,t):
    return bus13_seg1_t2[0]*100*m.Pg[18,t]*m.ug[18,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1_t2[2]* m.inertia_constant[18]*m.genD_Pmax[18]*m.ug[18,t]) + bus13_seg1_t2[3] + m.v13_5_t2[1,t]*A >= m.t13_5_t2[1,t]
m.bus13_5_u1_t2 = Constraint(m.PERIOD, rule = bus13_5_u1_t2)

def bus13_5_l2_t2(m,t):
    return bus13_seg2_t2[0]*100*m.Pg[18,t]*m.ug[18,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2_t2[2]* m.inertia_constant[18]*m.genD_Pmax[18]*m.ug[18,t]) + bus13_seg2_t2[3] <= m.t13_5_t2[1,t]
m.bus13_5_l2_t2 = Constraint(m.PERIOD,rule = bus13_5_l2_t2)

def bus13_5_u2_t2(m,t):
    return bus13_seg2_t2[0]*100*m.Pg[18,t]*m.ug[18,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2_t2[2]* m.inertia_constant[18]*m.genD_Pmax[18]*m.ug[18,t]) + bus13_seg2_t2[3] + (1 - m.v13_5_t2[1,t])*A >= m.t13_5_t2[1,t]
m.bus13_5_u2_t2 = Constraint(m.PERIOD,rule = bus13_5_u2_t2)

def bus13_5_t12_t2(m,t):
    return m.t13_5_t2[1,t] <= m.t13_5_t2[2,t]
m.bus13_5_t12_t2 = Constraint(m.PERIOD,rule = bus13_5_t12_t2)

def bus13_5_t21_t2(m,t):
    return m.t13_5_t2[2,t]<=m.t13_5_t2[1,t] + m.v13_5_t2[2,t]*A
m.bus13_5_t21_t2 = Constraint(m.PERIOD,rule = bus13_5_t21_t2)

def bus13_5_l3_t2(m,t):
    return bus13_seg3_t2[0]*100*m.Pg[18,t]*m.ug[18,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3_t2[2]* m.inertia_constant[18]*m.genD_Pmax[18]*m.ug[18,t]) + bus13_seg3_t2[3] <= m.t13_5_t2[2,t]
m.bus13_5_l3_t2 = Constraint(m.PERIOD,rule = bus13_5_l3_t2)

def bus13_5_u3_t2(m,t):
    return bus13_seg3_t2[0]*100*m.Pg[18,t]*m.ug[18,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3_t2[2]* m.inertia_constant[18]*m.genD_Pmax[18]*m.ug[18,t]) + bus13_seg3_t2[3] + (1 - m.v13_5_t2[2,t])*A >= m.t13_5_t2[2,t]
m.bus13_5_u3_t2 = Constraint(m.PERIOD,rule = bus13_5_u3_t2)

def bus13_5_t23_t2(m,t):
    return m.t13_5_t2[2,t] <= m.t13_5_t2[3,t]
m.bus13_5_t23_t2 = Constraint(m.PERIOD,rule = bus13_5_t23_t2)

def bus13_5_t32_t2(m,t):
    return m.t13_5_t2[3,t]<=m.t13_5_t2[2,t] + m.v13_5_t2[3,t]*A
m.bus13_5_t32_t2 = Constraint(m.PERIOD,rule = bus13_5_t32_t2)

def bus13_5_l4_t2(m,t):
    return bus13_seg4_t2[0]*100*m.Pg[18,t]*m.ug[18,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4_t2[2]* m.inertia_constant[18]*m.genD_Pmax[18]*m.ug[18,t]) + bus13_seg4_t2[3] <= m.t13_5_t2[3,t]
m.bus13_5_l4_t2 = Constraint(m.PERIOD,rule = bus13_5_l4_t2)

def bus13_5_u4_t2(m,t):
    return bus13_seg4_t2[0]*100*m.Pg[18,t]*m.ug[18,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4_t2[2]* m.inertia_constant[18]*m.genD_Pmax[18]*m.ug[18,t]) + bus13_seg4_t2[3] + (1 - m.v13_5_t2[3,t])*A >= m.t13_5_t2[3,t]
m.bus13_5_u4_t2 = Constraint(m.PERIOD,rule = bus13_5_u4_t2)

def bus13_6_R_t2(m,t):
    return m.t13_6_t2[3,t] <= RoCoF
m.bus13_6_R_t2 = Constraint(m.PERIOD, rule = bus13_6_R_t2)

def bus13_6_l1_t2(m,t):
    return bus13_seg1_t2[0]*100*m.Pg[19,t]*m.ug[19,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1_t2[2]* m.inertia_constant[19]*m.genD_Pmax[19]*m.ug[19,t]) + bus13_seg1_t2[3] <= m.t13_6_t2[1,t]
m.bus13_6_l1_t2 = Constraint(m.PERIOD, rule = bus13_6_l1_t2)

def bus13_6_u1_t2(m,t):
    return bus13_seg1_t2[0]*100*m.Pg[19,t]*m.ug[19,t] + bus13_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg1_t2[2]* m.inertia_constant[19]*m.genD_Pmax[19]*m.ug[19,t]) + bus13_seg1_t2[3] + m.v13_6_t2[1,t]*A >= m.t13_6_t2[1,t]
m.bus13_6_u1_t2 = Constraint(m.PERIOD, rule = bus13_6_u1_t2)

def bus13_6_l2_t2(m,t):
    return bus13_seg2_t2[0]*100*m.Pg[19,t]*m.ug[19,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2_t2[2]* m.inertia_constant[19]*m.genD_Pmax[19]*m.ug[19,t]) + bus13_seg2_t2[3] <= m.t13_6_t2[1,t]
m.bus13_6_l2_t2 = Constraint(m.PERIOD,rule = bus13_6_l2_t2)

def bus13_6_u2_t2(m,t):
    return bus13_seg2_t2[0]*100*m.Pg[19,t]*m.ug[19,t] + bus13_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg2_t2[2]* m.inertia_constant[19]*m.genD_Pmax[19]*m.ug[19,t]) + bus13_seg2_t2[3] + (1 - m.v13_6_t2[1,t])*A >= m.t13_6_t2[1,t]
m.bus13_6_u2_t2 = Constraint(m.PERIOD,rule = bus13_6_u2_t2)

def bus13_6_t12_t2(m,t):
    return m.t13_6_t2[1,t] <= m.t13_6_t2[2,t]
m.bus13_6_t12_t2 = Constraint(m.PERIOD,rule = bus13_6_t12_t2)

def bus13_6_t21_t2(m,t):
    return m.t13_6_t2[2,t]<=m.t13_6_t2[1,t] + m.v13_6_t2[2,t]*A
m.bus13_6_t21_t2 = Constraint(m.PERIOD,rule = bus13_6_t21_t2)

def bus13_6_l3_t2(m,t):
    return bus13_seg3_t2[0]*100*m.Pg[19,t]*m.ug[19,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3_t2[2]* m.inertia_constant[19]*m.genD_Pmax[19]*m.ug[19,t]) + bus13_seg3_t2[3] <= m.t13_6_t2[2,t]
m.bus13_6_l3_t2 = Constraint(m.PERIOD,rule = bus13_6_l3_t2)

def bus13_6_u3_t2(m,t):
    return bus13_seg3_t2[0]*100*m.Pg[19,t]*m.ug[19,t] + bus13_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg3_t2[2]* m.inertia_constant[19]*m.genD_Pmax[19]*m.ug[19,t]) + bus13_seg3_t2[3] + (1 - m.v13_6_t2[2,t])*A >= m.t13_6_t2[2,t]
m.bus13_6_u3_t2 = Constraint(m.PERIOD,rule = bus13_6_u3_t2)

def bus13_6_t23_t2(m,t):
    return m.t13_6_t2[2,t] <= m.t13_6_t2[3,t]
m.bus13_6_t23_t2 = Constraint(m.PERIOD,rule = bus13_6_t23_t2)

def bus13_6_t32_t2(m,t):
    return m.t13_6_t2[3,t]<=m.t13_6_t2[2,t] + m.v13_6_t2[3,t]*A
m.bus13_6_t32_t2 = Constraint(m.PERIOD,rule = bus13_6_t32_t2)

def bus13_6_l4_t2(m,t):
    return bus13_seg4_t2[0]*100*m.Pg[19,t]*m.ug[19,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4_t2[2]* m.inertia_constant[19]*m.genD_Pmax[19]*m.ug[19,t]) + bus13_seg4_t2[3] <= m.t13_6_t2[3,t]
m.bus13_6_l4_t2 = Constraint(m.PERIOD,rule = bus13_6_l4_t2)

def bus13_6_u4_t2(m,t):
    return bus13_seg4_t2[0]*100*m.Pg[19,t]*m.ug[19,t] + bus13_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus13_seg4_t2[2]* m.inertia_constant[19]*m.genD_Pmax[19]*m.ug[19,t]) + bus13_seg4_t2[3] + (1 - m.v13_6_t2[3,t])*A >= m.t13_6_t2[3,t]
m.bus13_6_u4_t2 = Constraint(m.PERIOD,rule = bus13_6_u4_t2)

# Node 15 locational RoCoF constraints

def bus15_R_t2(m,g,t):
    if g >= 20:
        if g <= 24:
            return m.t15_t2[3,t] <= RoCoF
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_R_t2 = Constraint(m.GEND, m.PERIOD,rule = bus15_R_t2)

def bus15_l1_t2(m,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg1_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg1_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg1_t2[3]<= m.t15_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_l1_t2 = Constraint(m.GEND,m.PERIOD,rule = bus15_l1_t2)

def bus15_u1_t2(m,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg1_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg1_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg1_t2[3] + m.v15_t2[1,t]*A >= m.t15_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_u1_t2 = Constraint(m.GEND, m.PERIOD, rule = bus15_u1_t2)

def bus15_l2_t2(m,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg2_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg2_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg2_t2[3] <= m.t15_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_l2_t2 = Constraint(m.GEND, m.PERIOD,rule = bus15_l2_t2)

def bus15_u2_t2(m,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg2_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg2_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg2_t2[3] + ( 1 - m.v15_t2[1, t]) * A >= m.t15_t2[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_u2_t2 = Constraint(m.GEND, m.PERIOD, rule = bus15_u2_t2)

def bus15_t12_t2(m,g,t):
    if g >= 20:
        if g <= 24:
            return m.t15_t2[1,t] <= m.t15_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_t12_t2 = Constraint(m.GEND, m.PERIOD,rule = bus15_t12_t2)

def bus15_t21_t2(m,g,t):
    if g >= 20:
        if g <= 24:
            return m.t15_t2[2,t]<=m.t15_t2[1,t] + m.v15_t2[2,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_t21_t2 = Constraint(m.GEND,m.PERIOD, rule = bus15_t21_t2)

def bus15_l3_t2(m,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg3_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg3_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg3_t2[3] <= m.t15_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_l3_t2 = Constraint(m.GEND,m.PERIOD, rule = bus15_l3_t2)

def bus15_u3_t2(m,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg3_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg3_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg3_t2[3] + (1 - m.v15_t2[2,t])*A >= m.t15_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_u3_t2 = Constraint(m.GEND, m.PERIOD,rule = bus15_u3_t2)

def bus15_t23_t2(m,g,t):
    if g >= 20:
        if g <= 24:
            return m.t15_t2[2,t] <= m.t15_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_t23_t2 = Constraint(m.GEND,m.PERIOD, rule = bus15_t23_t2)

def bus15_t32_t2(m,g,t):
    if g >= 20:
        if g <= 24:
            return m.t15_t2[3,t]<=m.t15_t2[2,t] + m.v15_t2[3,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_t32_t2 = Constraint(m.GEND, m.PERIOD,rule = bus15_t32_t2)

def bus15_l4_t2(m,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg4_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg4_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg4_t2[3] <= m.t15_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_l4_t2 = Constraint(m.GEND,m.PERIOD,rule = bus15_l4_t2)

def bus15_u4_t2(m,g,t):
    if g >= 20:
        if g <= 24:
            return bus15_seg4_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg4_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg4_t2[3] + (1 - m.v15_t2[3,t])*A >= m.t15_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_u4_t2 = Constraint(m.GEND,m.PERIOD,rule = bus15_u4_t2)

# Node 15_1 constraint
def bus15_1_R_t2(m,g,t):
    if g >= 25:
        if g <= 26:
            return m.t15_1_t2[3,t] <= RoCoF
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_R_t2 = Constraint(m.GEND, m.PERIOD,rule = bus15_1_R_t2)

def bus15_1_l1_t2(m,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg1_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg1_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg1_t2[3] <= m.t15_1_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_l1_t2 = Constraint(m.GEND,m.PERIOD,rule = bus15_1_l1_t2)

def bus15_1_u1_t2(m,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg1_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg1_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg1_t2[3]+ m.v15_1_t2[1,t]*A >= m.t15_1_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_u1_t2 = Constraint(m.GEND, m.PERIOD, rule = bus15_1_u1_t2)

def bus15_1_l2_t2(m,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg2_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg2_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg2_t2[3] <= m.t15_1_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_l2_t2 = Constraint(m.GEND, m.PERIOD,rule = bus15_1_l2_t2)

def bus15_1_u2_t2(m,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg2_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg2_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg2_t2[3] + ( 1 - m.v15_1_t2[1, t]) * A >= m.t15_1_t2[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_u2_t2 = Constraint(m.GEND, m.PERIOD, rule = bus15_1_u2_t2)

def bus15_1_t12_t2(m,g,t):
    if g >= 25:
        if g <= 26:
            return m.t15_1_t2[1,t] <= m.t15_1_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_t12_t2 = Constraint(m.GEND, m.PERIOD,rule = bus15_1_t12_t2)

def bus15_1_t21_t2(m,g,t):
    if g >= 25:
        if g <= 26:
            return m.t15_1_t2[2,t]<=m.t15_1_t2[1,t] + m.v15_1_t2[2,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_t21_t2 = Constraint(m.GEND,m.PERIOD, rule = bus15_1_t21_t2)

def bus15_1_l3_t2(m,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg3_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg3_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg3_t2[3] <= m.t15_1_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_l3_t2 = Constraint(m.GEND,m.PERIOD, rule = bus15_1_l3_t2)

def bus15_1_u3_t2(m,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg3_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg3_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg3_t2[3] + (1 - m.v15_1_t2[2,t])*A >= m.t15_1_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_u3_t2 = Constraint(m.GEND, m.PERIOD,rule = bus15_1_u3_t2)

def bus15_1_t23_t2(m,g,t):
    if g >= 25:
        if g <= 26:
            return m.t15_1_t2[2,t] <= m.t15_1_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_t23_t2 = Constraint(m.GEND,m.PERIOD, rule = bus15_1_t23_t2)

def bus15_1_t32_t2(m,g,t):
    if g >= 25:
        if g <= 26:
            return m.t15_1_t2[3,t]<=m.t15_1_t2[2,t] + m.v15_1_t2[3,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_t32_t2 = Constraint(m.GEND, m.PERIOD,rule = bus15_1_t32_t2)

def bus15_1_l4_t2(m,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg4_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg4_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg4_t2[3] <= m.t15_1_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_l4_t2 = Constraint(m.GEND,m.PERIOD,rule = bus15_1_l4_t2)

def bus15_1_u4_t2(m,g,t):
    if g >= 25:
        if g <= 26:
            return bus15_seg4_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus15_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus15_seg4_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus15_seg4_t2[3] + (1 - m.v15_1_t2[3,t])*A >= m.t15_1_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus15_1_u4_t2 = Constraint(m.GEND,m.PERIOD,rule = bus15_1_u4_t2)

# Node 16 locational RoCoF constraints

def bus16_R_t2(m,g,t):
    if g >= 27:
        if g <= 29:
            return m.t16_t2[3,t] <= RoCoF
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_R_t2 = Constraint(m.GEND, m.PERIOD,rule = bus16_R_t2)

def bus16_l1_t2(m,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg1_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus16_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus16_seg1_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus16_seg1_t2[3] <= m.t16_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_l1_t2 = Constraint(m.GEND,m.PERIOD,rule = bus16_l1_t2)

def bus16_u1_t2(m,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg1_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus16_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus16_seg1_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus16_seg1_t2[3]+ m.v16_t2[1,t]*A >= m.t16_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_u1_t2 = Constraint(m.GEND, m.PERIOD, rule = bus16_u1_t2)

def bus16_l2_t2(m,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg2_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus16_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus16_seg2_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus16_seg2_t2[3]<= m.t16_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_l2_t2 = Constraint(m.GEND, m.PERIOD,rule = bus16_l2_t2)

def bus16_u2_t2(m,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg2_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus16_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus16_seg2_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus16_seg2_t2[3] + ( 1 - m.v16_t2[1, t]) * A >= m.t16_t2[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_u2_t2 = Constraint(m.GEND, m.PERIOD, rule = bus16_u2_t2)

def bus16_t12_t2(m,g,t):
    if g >= 27:
        if g <= 29:
            return m.t16_t2[1,t] <= m.t16_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_t12_t2 = Constraint(m.GEND, m.PERIOD,rule = bus16_t12_t2)

def bus16_t21_t2(m,g,t):
    if g >= 27:
        if g <= 29:
            return m.t16_t2[2,t]<=m.t16_t2[1,t] + m.v16[2,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_t21_t2 = Constraint(m.GEND,m.PERIOD, rule = bus16_t21_t2)

def bus16_l3_t2(m,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg3_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus16_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus16_seg3_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus16_seg3_t2[3] <= m.t16_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_l3_t2 = Constraint(m.GEND,m.PERIOD, rule = bus16_l3_t2)

def bus16_u3_t2(m,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg3_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus16_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus16_seg3_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus16_seg3_t2[3] + (1 - m.v16_t2[2,t])*A >= m.t16_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_u3_t2 = Constraint(m.GEND, m.PERIOD,rule = bus16_u3_t2)

def bus16_t23_t2(m,g,t):
    if g >= 27:
        if g <= 29:
            return m.t16_t2[2,t] <= m.t16_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_t23_t2 = Constraint(m.GEND,m.PERIOD, rule = bus16_t23_t2)

def bus16_t32_t2(m,g,t):
    if g >= 27:
        if g <= 29:
            return m.t16_t2[3,t]<=m.t16_t2[2,t] + m.v16_t2[3,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_t32_t2 = Constraint(m.GEND, m.PERIOD,rule = bus16_t32_t2)

def bus16_l4_t2(m,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg4_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus16_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus16_seg4_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus16_seg4_t2[3] <= m.t16_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_l4_t2 = Constraint(m.GEND,m.PERIOD,rule = bus16_l4_t2)

def bus16_u4_t2(m,g,t):
    if g >= 27:
        if g <= 29:
            return bus16_seg4_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus16_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus16_seg4_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus16_seg4_t2[3] + (1 - m.v16_t2[3,t])*A >= m.t16_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus16_u4_t2 = Constraint(m.GEND,m.PERIOD,rule = bus16_u4_t2)

# Node 18 locational RoCoF constraints

def bus18_R_t2(m,t):
    return m.t18_t2[3,t] <= RoCoF
m.bus18_R_t2 = Constraint(m.PERIOD, rule = bus18_R_t2)

def bus18_l1_t2(m,t):
    return bus18_seg1_t2[0]*100*m.Pg[30,t]*m.ug[30,t] + bus18_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus18_seg1_t2[2]* m.inertia_constant[30]*m.genD_Pmax[30]*m.ug[30,t]) + bus18_seg1_t2[3] <= m.t18_t2[1,t]
m.bus18_l1_t2 = Constraint(m.PERIOD, rule = bus18_l1_t2)

def bus18_u1_t2(m,t):
    return bus18_seg1_t2[0]*100*m.Pg[30,t]*m.ug[30,t] + bus18_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus18_seg1_t2[2]* m.inertia_constant[30]*m.genD_Pmax[30]*m.ug[30,t]) + bus18_seg1_t2[3] + m.v18_t2[1,t]*A >= m.t18_t2[1,t]
m.bus18_u1_t2 = Constraint(m.PERIOD, rule = bus18_u1_t2)

def bus18_l2_t2(m,t):
    return bus18_seg2_t2[0]*100*m.Pg[30,t]*m.ug[30,t] + bus18_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus18_seg2_t2[2]* m.inertia_constant[30]*m.genD_Pmax[30]*m.ug[30,t]) + bus18_seg2_t2[3] <= m.t18_t2[1,t]
m.bus18_l2_t2 = Constraint(m.PERIOD,rule = bus18_l2_t2)

def bus18_u2_t2(m,t):
    return bus18_seg2_t2[0]*100*m.Pg[30,t]*m.ug[30,t] + bus18_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus18_seg2_t2[2]* m.inertia_constant[30]*m.genD_Pmax[30]*m.ug[30,t]) + bus18_seg2_t2[3]+ (1 - m.v18_t2[1,t])*A >= m.t18_t2[1,t]
m.bus18_u2_t2 = Constraint(m.PERIOD,rule = bus18_u2_t2)

def bus18_t12_t2(m,t):
    return m.t18_t2[1,t] <= m.t18_t2[2,t]
m.bus18_t12_t2 = Constraint(m.PERIOD,rule = bus18_t12_t2)

def bus18_t21_t2(m,t):
    return m.t18_t2[2,t]<=m.t18_t2[1,t] + m.v18_t2[2,t]*A
m.bus18_t21_t2 = Constraint(m.PERIOD,rule = bus18_t21_t2)

def bus18_l3_t2(m,t):
    return bus18_seg3_t2[0]*100*m.Pg[30,t]*m.ug[30,t] + bus18_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus18_seg3_t2[2]* m.inertia_constant[30]*m.genD_Pmax[30]*m.ug[30,t]) + bus18_seg3_t2[3]<= m.t18_t2[2,t]
m.bus18_l3_t2 = Constraint(m.PERIOD,rule = bus18_l3_t2)

def bus18_u3_t2(m,t):
    return bus18_seg3_t2[0]*100*m.Pg[30,t]*m.ug[30,t] + bus18_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus18_seg3_t2[2]* m.inertia_constant[30]*m.genD_Pmax[30]*m.ug[30,t]) + bus18_seg3_t2[3] + (1 - m.v18_t2[2,t])*A >= m.t18_t2[2,t]
m.bus18_u3_t2 = Constraint(m.PERIOD,rule = bus18_u3_t2)

def bus18_t23_t2(m,t):
    return m.t18_t2[2,t] <= m.t18_t2[3,t]
m.bus18_t23_t2 = Constraint(m.PERIOD,rule = bus18_t23_t2)

def bus18_t32_t2(m,t):
    return m.t18_t2[3,t]<=m.t18_t2[2,t] + m.v18_t2[3,t]*A
m.bus18_t32_t2 = Constraint(m.PERIOD,rule = bus18_t32_t2)

def bus18_l4_t2(m,t):
    return bus18_seg4_t2[0]*100*m.Pg[30,t]*m.ug[30,t] + bus18_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus18_seg4_t2[2]* m.inertia_constant[30]*m.genD_Pmax[30]*m.ug[30,t]) + bus18_seg4_t2[3] <= m.t18_t2[3,t]
m.bus18_l4_t2 = Constraint(m.PERIOD,rule = bus18_l4_t2)

def bus18_u4_t2(m,t):
    return bus18_seg4_t2[0]*100*m.Pg[30,t]*m.ug[30,t] + bus18_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus18_seg4_t2[2]* m.inertia_constant[30]*m.genD_Pmax[30]*m.ug[30,t]) + bus18_seg4_t2[3] + (1 - m.v18_t2[3,t])*A >= m.t18_t2[3,t]
m.bus18_u4_t2 = Constraint(m.PERIOD,rule = bus18_u4_t2)

# Node 21 locational RoCoF constraints

def bus21_R_t2(m,g,t):
    if g >= 31:
        if g <= 35:
            return m.t21_t2[3,t] <= RoCoF
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_R_t2 = Constraint(m.GEND, m.PERIOD,rule = bus21_R_t2)

def bus21_l1_t2(m,g,t):
    if g >= 31:
        if g <= 35:
            return bus21_seg1_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus21_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus21_seg1_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus21_seg1_t2[3] <= m.t21_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_l1_t2 = Constraint(m.GEND,m.PERIOD,rule = bus21_l1_t2)

def bus21_u1_t2(m,g,t):
    if g >= 31:
        if g <= 35:
            return bus21_seg1_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus21_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus21_seg1_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus21_seg1_t2[3] + m.v21_t2[1,t]*A >= m.t21_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_u1_t2 = Constraint(m.GEND, m.PERIOD, rule = bus21_u1_t2)

def bus21_l2_t2(m,g,t):
    if g >= 31:
        if g <= 35:
            return bus21_seg2_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus21_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus21_seg2_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus21_seg2_t2[3] <= m.t21_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_l2_t2 = Constraint(m.GEND, m.PERIOD,rule = bus21_l2_t2)

def bus21_u2_t2(m,g,t):
    if g >= 31:
        if g <= 35:
            return bus21_seg2_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus21_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus21_seg2_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus21_seg2_t2[3] + ( 1 - m.v21_t2[1, t]) * A >= m.t21_t2[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_u2_t2 = Constraint(m.GEND, m.PERIOD, rule = bus21_u2_t2)

def bus21_t12_t2(m,g,t):
    if g >= 31:
        if g <= 35:
            return m.t21_t2[1,t] <= m.t21_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_t12_t2 = Constraint(m.GEND, m.PERIOD,rule = bus21_t12_t2)

def bus21_t21_t2(m,g,t):
    if g >= 31:
        if g <= 35:
            return m.t21_t2[2,t]<=m.t21_t2[1,t] + m.v21_t2[2,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_t21_t2 = Constraint(m.GEND,m.PERIOD, rule = bus21_t21_t2)

def bus21_l3_t2(m,g,t):
    if g >= 31:
        if g <= 35:
            return bus21_seg3_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus21_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus21_seg3_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus21_seg3_t2[3] <= m.t21_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_l3_t2 = Constraint(m.GEND,m.PERIOD, rule = bus21_l3_t2)

def bus21_u3_t2(m,g,t):
    if g >= 31:
        if g <= 35:
            return bus21_seg3_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus21_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus21_seg3_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus21_seg3_t2[3] + (1 - m.v21_t2[2,t])*A >= m.t21_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_u3_t2 = Constraint(m.GEND, m.PERIOD,rule = bus21_u3_t2)

def bus21_t23_t2(m,g,t):
    if g >= 31:
        if g <= 35:
            return m.t21_t2[2,t] <= m.t21_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_t23_t2 = Constraint(m.GEND,m.PERIOD, rule = bus21_t23_t2)

def bus21_t32_t2(m,g,t):
    if g >= 31:
        if g <= 35:
            return m.t21_t2[3,t]<=m.t21_t2[2,t] + m.v21_t2[3,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_t32_t2 = Constraint(m.GEND, m.PERIOD,rule = bus21_t32_t2)

def bus21_l4_t2(m,g,t):
    if g >= 31:
        if g <= 35:
            return bus21_seg4_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus21_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus21_seg4_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus21_seg4_t2[3] <= m.t21_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_l4_t2 = Constraint(m.GEND,m.PERIOD,rule = bus21_l4_t2)

def bus21_u4_t2(m,g,t):
    if g >= 31:
        if g <= 35:
            return bus21_seg4_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus21_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus21_seg4_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus21_seg4_t2[3] + (1 - m.v21_t2[3,t])*A >= m.t21_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus21_u4_t2 = Constraint(m.GEND,m.PERIOD,rule = bus21_u4_t2)

# Node 22 constraints
def bus22_R_t2(m,t):
    return m.t22_t2[3,t] <= RoCoF
m.bus22_R_t2 = Constraint(m.PERIOD, rule = bus22_R_t2)

def bus22_l1_t2(m,t):
    return bus22_seg1_t2[0]*100*m.Pg[36,t]*m.ug[36,t] + bus22_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus22_seg1_t2[2]* m.inertia_constant[36]*m.genD_Pmax[36]*m.ug[36,t]) + bus22_seg1_t2[3] <= m.t22_t2[1,t]
m.bus22_l1_t2 = Constraint(m.PERIOD, rule = bus22_l1_t2)

def bus22_u1_t2(m,t):
    return bus22_seg1_t2[0]*100*m.Pg[36,t]*m.ug[36,t] + bus22_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus22_seg1_t2[2]* m.inertia_constant[36]*m.genD_Pmax[36]*m.ug[36,t]) + bus22_seg1_t2[3] + m.v22_t2[1,t]*A >= m.t22_t2[1,t]
m.bus22_u1_t2 = Constraint(m.PERIOD, rule = bus22_u1_t2)

def bus22_l2_t2(m,t):
    return bus22_seg2_t2[0]*100*m.Pg[36,t]*m.ug[36,t] + bus22_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus22_seg2_t2[2]* m.inertia_constant[36]*m.genD_Pmax[36]*m.ug[36,t]) + bus22_seg2_t2[3] <= m.t22_t2[1,t]
m.bus22_l2_t2 = Constraint(m.PERIOD,rule = bus22_l2_t2)

def bus22_u2_t2(m,t):
    return bus22_seg2_t2[0]*100*m.Pg[36,t]*m.ug[36,t] + bus22_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus22_seg2_t2[2]* m.inertia_constant[36]*m.genD_Pmax[36]*m.ug[36,t]) + bus22_seg2_t2[3] + (1 - m.v22_t2[1,t])*A >= m.t22_t2[1,t]
m.bus22_u2_t2 = Constraint(m.PERIOD,rule = bus22_u2_t2)

def bus22_t12_t2(m,t):
    return m.t22_t2[1,t] <= m.t22_t2[2,t]
m.bus22_t12_t2 = Constraint(m.PERIOD,rule = bus22_t12_t2)

def bus22_t21_t2(m,t):
    return m.t22_t2[2,t]<=m.t22_t2[1,t] + m.v22_t2[2,t]*A
m.bus22_t21_t2 = Constraint(m.PERIOD,rule = bus22_t21_t2)

def bus22_l3_t2(m,t):
    return bus22_seg3_t2[0]*100*m.Pg[36,t]*m.ug[36,t] + bus22_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus22_seg3_t2[2]* m.inertia_constant[36]*m.genD_Pmax[36]*m.ug[36,t]) + bus22_seg3_t2[3] <= m.t22_t2[2,t]
m.bus22_l3_t2 = Constraint(m.PERIOD,rule = bus22_l3_t2)

def bus22_u3_t2(m,t):
    return bus22_seg3_t2[0]*100*m.Pg[36,t]*m.ug[36,t] + bus22_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus22_seg3_t2[2]* m.inertia_constant[36]*m.genD_Pmax[36]*m.ug[36,t]) + bus22_seg3_t2[3] + (1 - m.v22_t2[2,t])*A >= m.t22_t2[2,t]
m.bus22_u3_t2 = Constraint(m.PERIOD,rule = bus22_u3_t2)

def bus22_t23_t2(m,t):
    return m.t22_t2[2,t] <= m.t22_t2[3,t]
m.bus22_t23_t2 = Constraint(m.PERIOD,rule = bus22_t23_t2)

def bus22_t32_t2(m,t):
    return m.t22_t2[3,t]<=m.t22_t2[2,t] + m.v22_t2[3,t]*A
m.bus22_t32_t2 = Constraint(m.PERIOD,rule = bus22_t32_t2)

def bus22_l4_t2(m,t):
    return bus22_seg4_t2[0]*100*m.Pg[36,t]*m.ug[36,t] + bus22_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus22_seg4_t2[2]* m.inertia_constant[36]*m.genD_Pmax[36]*m.ug[36,t]) + bus22_seg4_t2[3] <= m.t22_t2[3,t]
m.bus22_l4_t2 = Constraint(m.PERIOD,rule = bus22_l4_t2)

def bus22_u4_t2(m,t):
    return bus22_seg4_t2[0]*100*m.Pg[36,t]*m.ug[36,t] + bus22_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus22_seg4_t2[2]* m.inertia_constant[36]*m.genD_Pmax[36]*m.ug[36,t]) + bus22_seg4_t2[3] + (1 - m.v22_t2[3,t])*A >= m.t22_t2[3,t]
m.bus22_u4_t2 = Constraint(m.PERIOD,rule = bus22_u4_t2)


# Node 23 locational RoCoF constraints

def bus23_R_t2(m,g,t):
    if g >= 39:
        if g <= 41:
            return m.t23_t2[3,t] <= RoCoF
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_R_t2 = Constraint(m.GEND, m.PERIOD,rule = bus23_R_t2)

def bus23_l1_t2(m,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg1_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus23_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus23_seg1_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus23_seg1_t2[3]  <= m.t23_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_l1_t2 = Constraint(m.GEND,m.PERIOD,rule = bus23_l1_t2)

def bus23_u1_t2(m,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg1_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus23_seg1_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus23_seg1_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus23_seg1_t2[3]  + m.v23_t2[1,t]*A >= m.t23_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_u1_t2 = Constraint(m.GEND, m.PERIOD, rule = bus23_u1_t2)

def bus23_l2_t2(m,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg2_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus23_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus23_seg2_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus23_seg2_t2[3]  <= m.t23_t2[1,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_l2_t2 = Constraint(m.GEND, m.PERIOD,rule = bus23_l2_t2)

def bus23_u2_t2(m,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg2_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus23_seg2_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus23_seg2_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus23_seg2_t2[3]  + ( 1 - m.v23_t2[1, t]) * A >= m.t23_t2[1, t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_u2_t2 = Constraint(m.GEND, m.PERIOD, rule = bus23_u2_t2)

def bus23_t12_t2(m,g,t):
    if g >= 39:
        if g <= 41:
            return m.t23_t2[1,t] <= m.t23_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_t12_t2 = Constraint(m.GEND, m.PERIOD,rule = bus23_t12_t2)

def bus23_t21_t2(m,g,t):
    if g >= 39:
        if g <= 41:
            return m.t23_t2[2,t]<=m.t23_t2[1,t] + m.v23_t2[2,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_t21_t2 = Constraint(m.GEND,m.PERIOD, rule = bus23_t21_t2)

def bus23_l3_t2(m,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg3_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus23_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus23_seg3_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus23_seg3_t2[3]  <= m.t23_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_l3_t2 = Constraint(m.GEND,m.PERIOD, rule = bus23_l3_t2)

def bus23_u3_t2(m,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg3_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus23_seg3_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus23_seg3_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus23_seg3_t2[3]  + (1 - m.v23_t2[2,t])*A >= m.t23_t2[2,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_u3_t2 = Constraint(m.GEND, m.PERIOD,rule = bus23_u3_t2)

def bus23_t23_t2(m,g,t):
    if g >= 39:
        if g <= 41:
            return m.t23_t2[2,t] <= m.t23_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_t23_t2 = Constraint(m.GEND,m.PERIOD, rule = bus23_t23_t2)

def bus23_t32_t2(m,g,t):
    if g >= 39:
        if g <= 41:
            return m.t23_t2[3,t]<=m.t23_t2[2,t] + m.v23_t2[3,t]*A
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_t32_t2 = Constraint(m.GEND, m.PERIOD,rule = bus23_t32_t2)

def bus23_l4_t2(m,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg4_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus23_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus23_seg4_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus23_seg4_t2[3]  <= m.t23_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_l4_t2 = Constraint(m.GEND,m.PERIOD,rule = bus23_l4_t2)

def bus23_u4_t2(m,g,t):
    if g >= 39:
        if g <= 41:
            return bus23_seg4_t2[0]*100*m.Pg[g,t]*m.ug[g,t] + bus23_seg4_t2[1]/(fn*np.pi)*(sum(m.inertia_constant[j]*m.genD_Pmax[j]*m.ug[j,t] for j in m.GEND) + bus23_seg4_t2[2]*m.inertia_constant[g]*m.genD_Pmax[g]*m.ug[g,t]) +bus23_seg4_t2[3]  + (1 - m.v23_t2[3,t])*A >= m.t23_t2[3,t]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
m.bus23_u4_t2 = Constraint(m.GEND,m.PERIOD,rule = bus23_u4_t2)

m.dual = pyomo.environ.Suffix(direction=pyomo.environ.Suffix.IMPORT_EXPORT)
instance2 = m.create_instance('./dataFile24BusAllinertia41sen_4EP.dat')

for t in range(23):
    instance2.theta[13,t+1].fixed = True
    instance2.theta[13,t+1].value = 0

for i in instance2.GEND:
    for j in instance2.PERIOD:
        instance2.ug[i,j] = int(instance.ug[i,j]())
for i in instance2.GEND:
    for j in instance2.PERIOD:
        instance2.vg[i,j]= int(instance.vg[i,j]())
for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v01[i,j]= int(instance.v01[i,j]())
for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v02_1[i,j]= int(instance.v02_1[i,j]())
for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v02_2[i,j]= int(instance.v02_2[i,j]())
for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v02_3[i,j]= int(instance.v02_3[i,j]())
for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v02_3[i,j]= int(instance.v02_3[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v02_4[i,j]= int(instance.v02_4[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v02_5[i,j]= int(instance.v02_5[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v02_6[i,j]= int(instance.v02_6[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v07_1[i,j]= int(instance.v07_1[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v07_2[i,j]= int(instance.v07_2[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v07_3[i,j]= int(instance.v07_3[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v13_1[i,j]= int(instance.v13_1[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v13_2[i,j]= int(instance.v13_2[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v13_3[i,j]= int(instance.v13_3[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v13_4[i,j]= int(instance.v13_4[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v13_5[i,j]= int(instance.v13_5[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v13_6[i,j]= int(instance.v13_6[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v15[i,j]= int(instance.v15[i,j]())
for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v15_1[i,j]= int(instance.v15_1[i,j]())
for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v16[i,j]= int(instance.v16[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v18[i,j]= int(instance.v18[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v21[i,j]= int(instance.v21[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v22[i,j]= int(instance.v22[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v23[i,j]= int(instance.v23[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v01_t2[i,j]= int(instance.v01_t2[i,j]())
for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v02_1_t2[i,j]= int(instance.v02_1_t2[i,j]())
for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v02_2_t2[i,j]= int(instance.v02_2_t2[i,j]())
for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v02_3_t2[i,j]= int(instance.v02_3_t2[i,j]())
for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v02_3_t2[i,j]= int(instance.v02_3_t2[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v02_4_t2[i,j]= int(instance.v02_4_t2[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v02_5_t2[i,j]= int(instance.v02_5_t2[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v02_6_t2[i,j]= int(instance.v02_6_t2[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v07_1_t2[i,j]= int(instance.v07_1_t2[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v07_2_t2[i,j]= int(instance.v07_2_t2[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v07_3_t2[i,j]= int(instance.v07_3_t2[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v13_1_t2[i,j]= int(instance.v13_1_t2[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v13_2_t2[i,j]= int(instance.v13_2_t2[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v13_3_t2[i,j]= int(instance.v13_3_t2[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v13_4_t2[i,j]= int(instance.v13_4_t2[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v13_5_t2[i,j]= int(instance.v13_5_t2[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v13_6_t2[i,j]= int(instance.v13_6_t2[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v15_t2[i,j]= int(instance.v15_t2[i,j]())
for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v15_1_t2[i,j]= int(instance.v15_1_t2[i,j]())
for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v16_t2[i,j]= int(instance.v16_t2[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v18_t2[i,j]= int(instance.v18_t2[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v21_t2[i,j]= int(instance.v21_t2[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v22_t2[i,j]= int(instance.v22_t2[i,j]())

for i in instance2.SEG:
    for j in instance2.PERIOD:
        instance2.v23_t2[i,j]= int(instance.v23_t2[i,j]())

SCUCsolver2 = SolverFactory('gurobi')
SCUCsolver2.options.mipgap = 0.0001
SCUCsolver2.solve(instance2).write()
results2 = SCUCsolver2.solve(instance2)
print("\nresults.Solution.Status: " + str(results2.Solution.Status))
print("\nresults.solver.status: " + str(results2.solver.status))
print("\nresults.solver.termination_condition: " + str(results2.solver.termination_condition))
print("\nresults.solver.termination_message: " + str(results2.solver.termination_message))
print('\nminimize cost: ' + str(instance2.obj()))

Cons = []
D = np.zeros((25,25))
for c in instance2.component_objects(pyomo.environ.Constraint, active =True):
    Cons.append(c)
cobject = getattr(instance2, str(Cons[0]))
for index in cobject:
    print(index)
    #print("   ", index, instance2.dual[cobject[index]])
    D[index] = instance2.dual[cobject[index]]
data = pd.DataFrame(D)
