# ---- Libraries ----#
import gurobipy as gp
import itertools
from gurobipy import GRB
import pandas as pd
import numpy as np

# ---- Read Files (Data) ---- #
UB = pd.read_csv('iJO1366_UB_n.csv',header=None)[0].tolist()

LB = pd.read_csv('iJO1366_LB_n.csv',header=None)[0].tolist()

rxn = pd.read_csv('iJO1366_rxn.csv',header=None)[0].tolist()

met = pd.read_csv('iJO1366_met.csv',header=None)[0].tolist()

c = pd.read_csv('iJO1366_c.csv',header=None)[0].tolist()

S = pd.read_csv('iJO1366_S.csv',header=None).values

r = pd.read_csv('iJO1366_r.csv',header=None)[0].tolist() # if reaction i is reversible r[i] = 1

names = pd.read_csv('iJO1366_names.csv',header=None)[0].tolist()

subs = pd.read_csv('iJO1366_Subsystem.csv',header=None)[0].tolist()

fba = pd.read_csv('iJO1366_WT.csv',header=None)[0].tolist()

# ---- Useful Lists --- #
b = [0 for i in range(len(met))]
reversible = []
irreversible = []

for index,value in enumerate(rxn):
    if r[index] == 1:
        reversible.append(index)
    else:
        irreversible.append(index)

set_reaction = ['GLCabcpp', 'GLCptspp', 'HEX1', 'PGI', 'PFK', 'FBA', 'TPI', 'GAPD','PGK', 'PGM', 'ENO', 'PYK',
'LDH_D', 'PFL', 'ALCD2x', 'PTAr', 'ACKr','G6PDH2r', 'PGL', 'GND', 'RPI', 'RPE', 'TKT1', 'TALA', 'TKT2', 'FUM',
'FRD2', 'SUCOAS', 'AKGDH', 'ACONTa', 'ACONTb', 'ICDHyr', 'CS', 'MDH','MDH2', 'MDH3', 'ACALD']

knockout = []

for i in set_reaction:
    knockout.append(rxn.index(i))

chemical = rxn.index('EX_succ_e')
biomass = rxn.index('BIOMASS_Ec_iJO1366_core_53p95M')
treshold = round(.5*fba[biomass],2)

N = [i for i in range(len(met))]
M = [i for i in range(len(rxn))]

k = 2

# ---- Gurobi Model ----#

# !!! N for metabolites; M for reactions !!!
x = gp.Model()

# Add variables v and y

v = x.addVars(M,lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='v') # v \in M
y = x.addVars(M,vtype=GRB.BINARY,name='y') # y \in M

# Add dual variables

l = x.addVars(N,lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='l') # lambda \in N
m = x.addVars(M,lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='m') # mu \in M
p = x.addVars(M,lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='p') # rho \in M
e = x.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY,name='e') #eta

# Add linearization variables

g = x.addVars(M,lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='g') # gamma \in M
o = x.addVars(M,lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='o') # omega \in M

# Set Objective

x.setObjective(1*v[chemical], GRB.MAXIMIZE)

# Dual Objective

x.addConstr(v[biomass] == (sum(b[i]*l[i] for i in N) +
                           sum(-LB[j]*g[j] for j in M) +
                           sum(UB[j]*o[j] for j in M) - treshold*e),name='Dual-Obj') #Dual Objective

# Stoichiometry Constraints

x.addConstrs((gp.quicksum(S[i,j] * v[j] for j in M) == 0 for i in N), name='St') #Stoichimetry


x.addConstrs((gp.quicksum(S.transpose()[i,j]*l[j] for j in N)
              - m[i]
              + p[i]
               == 0 for i in M if i !=biomass)
             ,name='S_dual')

x.addConstr((gp.quicksum(S.transpose()[biomass,j]*l[j] for j in N)
              - m[biomass]
              + p[biomass]
              - e
               == 1 for i in M if i == biomass)
             ,name='Sb_dual')

# Linearization

x.addConstrs((LB[j]*yj[j] <= g[j] for j in M),name='LB_g')
x.addConstrs((g[j] <= UB[j]*yj[j] for j in M),name='UB_g')

x.addConstrs((LB[j]*(1-y[j]) <= m[j] for j in M),name='lb-m')
x.addConstrs((m[j] <= UB[j]*(1-y[j])+g[j] for j in M),name='m-g')


x.addConstrs((LB[j]*yj[j] <= o[j] for j in M),name='LB_o')
x.addConstrs((o[j] <= UB[j]*yj[j] for j in M),name='UB_o')

x.addConstrs((LB[j]*(1-y[j]) <= p[j] for j in M),name='lb-p')
x.addConstrs((p[j] <= UB[j]*(1-y[j])+o[j] for j in M),name='p-o')

# LB <= v <= UB

x.addConstrs((LB[j]*y[j] <= v[j] for j in M), name='LBy')
x.addConstrs((v[j] <= UB[j]*y[j] for j in M), name='UBy')

# v_biomass >= treshold

x.addConstr(v[biomass] >= treshold,name='target')

# Knapsack

x.addConstr(sum(1-y[j] for j in knockout) == k, name='Knapsack')


x.write('iJO1366_Optknock_6.lp')
