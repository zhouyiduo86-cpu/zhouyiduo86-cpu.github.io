#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd

E_system = np.load("E_system_kgCO2.npy")
print("Loaded:", E_system.shape) 


# **What this model will answer:**  
#     “What is the expected economic impact and the worst economic impact of the action choice?”

# We already have:  
#     Emissions distribution from Monte Carlo

# $$
# E_i \sim \text{Monte Carlo}(\text{distance},\ \text{fuel efficiency},\ \text{load factors})
# $$
# 

# Define:  
# $$
# \text{Climate Cost} = E_i \times SSC
# $$

# $$ E_i = Emissions $$ 
# $$ SSC = social cost of carbon $$

# **scenario analysis:**

# Low SSC 
# 
# Central SSC  
# 
# High SSC  

# 
# by table:

# | type                  | case A | case B |
# |----------------------|------------|------------|
# | Expected climate cost | $X$        | $Y$        |
# | 95th percentile cost | $X_{95}$   | $Y_{95}$   |
# | Cost range            | $[min,\,max]$ | $[min,\,max]$ |
# 

# Choose a carbon price range in CAD:  
# From "Shadow Carbon Pricing in the Canadian Energy Sector"(https://institute.smartprosperity.ca/sites/default/files/publications/files/Shadow%20Carbon%20Pricing%20in%20the%20Canadian%20Energy%20Sector.pdf):  
# 
# Canadian firms commonly use C$15–C$68 / tCO₂  
# 
# Government of Canada SCC values used in regulation are C$26 / tCO₂ - ~C$104 / tCO₂

# | Case | Carbon price (CAD / tCO₂) |                                                  |
# |---------|----------------------------|--------------------------------------------------|
# | Low     |   C\$30                    | Conservative case        |
# | Central |   C\$65                    |  |
# | High    |   C\$100                   | Risk-averse case       |
# 

# Now:

# $$
# \text{Climate Cost} = E_i \times SSC
# $$

# $$ E_i = \text{tonnes CO}_2 \text{ per simulation draw} $$
# $$ \text{SSC}_{\text{CAD}} \in \{30,\ 65,\ 100\} $$

# In[12]:


import numpy as np
import pandas as pd

E_system = np.load("E_system_kgCO2.npy")
E_system_t = E_system / 1000
SSC_CAD = {
    "Low (C$30/tCO2)": 30,
    "Central (C$65/tCO2)": 65,
    "High (C$100/tCO2)": 100
}

def summarize(x):
    return {
        "Mean": float(np.mean(x)),
        "P5": float(np.quantile(x, 0.05)),
        "Median": float(np.quantile(x, 0.50)),
        "P95": float(np.quantile(x, 0.95))
    }

em_summary = summarize(E_system_t)
rows = []
for label, scc in SSC_CAD.items():
    cost = E_system_t * scc
    s = summarize(cost)
    rows.append([label, s["Mean"], s["P5"], s["Median"], s["P95"]])

ssc_results = pd.DataFrame(
    rows,
    columns=["SSC Scenario", "Mean (CAD)", "P5 (CAD)", "Median (CAD)", "P95 (CAD)"]
)
print("System emissions summary (tonnes CO2):", em_summary)
ssc_results


# Define:
# 
# $$
# PV_{\text{Carbon Cost},\, i}
# =
# E_i \times \sum_{t=1}^{T} \frac{SCC_{\text{annual}}}{(1+r)^t}
# $$
# 

# $$
# SCC_{\text{annual}} = \frac{SCC_{\text{total}}}{T}
# $$

# Where: 
# - Emissions occur today
# - Damages occur evenly over 𝑇 years
# - Discount rate 𝑟

# | Parameter | Choice | Reference |
# |---------|--------|-----------|
# | Horizon T | 50 years | GWP: FCCC/PA/CMA/2018/3/Add.2 |
# | Discount rate r | 2%–3% | En14-202-2016-eng |
# | SSC | 30 / 65 / 100 CAD | / |

# In[13]:


def pv_factor(T=50, r=0.03):
    return sum(1 / (1 + r)**t for t in range(1, T+1))


# In[14]:


T = 50
r = 0.03
discount_multiplier = pv_factor(T, r)
SSC_CAD = {
    "Low (30)": 30,
    "Central (65)": 65,
    "High (100)": 100
}

rows = []
for label, scc in SSC_CAD.items():
    scc_annual = scc / T
    pv_scc = scc_annual * discount_multiplier
    pv_cost = E_system_t * pv_scc
    rows.append([
        label,
        np.mean(pv_cost),
        np.quantile(pv_cost, 0.95)
    ])

pv_results = pd.DataFrame(
    rows,
    columns=["SSC scenario", "Mean PV cost (C$)", "P95 PV cost (C$)"]
)
pv_results


# In[ ]:




