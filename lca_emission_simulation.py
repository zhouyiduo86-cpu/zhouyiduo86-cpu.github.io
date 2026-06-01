#!/usr/bin/env python
# coding: utf-8

# **The Emissions Formula:**
# 
# For a LCA inventory, the total emission is calculated as:
# 
# $$
# M_{\text{total}} = \sum_{\text{i=0}} \left( \text{Fuel Comsumption} \right) \times \left( \text{$\text{EF}_{\text{CO}_2}$} \right)
# $$
# 
# where:
# 
# - $\text{Fuel Consumption}: L$
# - $\text{EF}_{\text{CO}_2}$: $\text{kg } \text{CO}_2/\text{L}$

# $$
# M_{\text{total}} 
# = 
# \left[
# \left( \frac{FE_{\text{long}}}{100} \times d_{\text{long}} \right)
# +
# \left( \frac{FE_{\text{short}}}{100} \times (p \times d_{\text{short}}) \right)
# \right]
# \times EF_{\text{CO}_2}
# $$
# 

# | **Model Parameters**                             | **Units**      | **Mean**      | **Uncertainty Distribution**                       | **Reference**                                                   |
# |---------------------------------------------------------|----------------|------------------------|-----------------------------------------------------|------------------------------------------------------------------|
# | Distance, Long-Haul (*d*<sub>long</sub>)                | km             | 3624                   | N/A                                 | *Dlong* data file                                         |
# | Distance, Short-Haul (*d*<sub>short</sub>)              | km             | 358.39                 | N/A                                 | *Dshort* data file                                        |
# | Total Mass Transported                                  | tonnes         | 8122.08                | N/A                                | *Dlong* data file                       |
# | CO₂ Emission Factor (*EF*<sub>CO₂</sub>)                | kg CO₂/L       | 2.68                   | N/A                                 | EN84-294-2025-eng.html (https://share.google/QiuBkVJ7IwWg8LO7e)               |
# | Fuel Economy, Long-Haul (*FE*<sub>long</sub>)           | L/100 km       | 38.00 (29 ~ 47 L/100km)        | Lognormal (μ<sub>log</sub>, σ<sub>log</sub>)        | https://www.fuelly.com/truck/international/4300/2017            |
# | Fuel Economy, Short-Haul (*FE*<sub>short</sub>)         | L/100 km       | 27.00        | Lognormal (μ<sub>log</sub>, σ<sub>log</sub>)        | https://www.fuelly.com/truck/international/4300/2017             |
# | Pavement–Vehicle Interaction Factor (*p*)               | N/A       | 1.3                    | Triangular (min = 1.0, mode = 1.3, max = 1.5)       | https://www.sciencedirect.com/journal/resources-conservation-and-recycling    |
# 

# ***Deterministic LCA***

# excel

# ***Probabilistic LCA***

# In[11]:


pip install SALib


# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "/Users/a20560/Desktop/backgroundss/useful/Data from Toronto.xlsx"

df_long  = pd.read_excel(file_path, sheet_name="Dlong",  header=1)
df_short = pd.read_excel(file_path, sheet_name="Dshort", header=0)

df_long.columns  = df_long.columns.astype(str).str.strip()
df_short.columns = df_short.columns.astype(str).str.strip()

def num_series(df, col):
    return pd.to_numeric(df[col], errors="coerce").dropna()

s_local = num_series(df_short, "Total Distance(km)")
s_long  = num_series(df_long,  "Distance Long (dlong)")

s_w = (num_series(df_long, "Weight (Assume Woodchip Density (160 kg/m3))") / 1000.0)

p_default = 0.50
s_p = pd.Series(np.full(len(s_long), p_default), name="share_long_p")

p_col_candidates = [c for c in df_long.columns if "share" in c.lower() and "long" in c.lower()]
if len(p_col_candidates) > 0:
    s_p = pd.to_numeric(df_long[p_col_candidates[0]], errors="coerce").dropna()
    if s_p.max() > 1:
        s_p = s_p / 100.0

EF_t_mean = 0.06
EF_t_sd   = 0.01
rng = np.random.default_rng(42)
s_EFt = rng.normal(EF_t_mean, EF_t_sd, size=1000)
s_EFt = np.clip(s_EFt, 0, None)

def summarize(name, s):
    s = np.asarray(s)
    return pd.Series({
        "count": len(s),
        "mean": float(np.mean(s)),
        "std": float(np.std(s, ddof=1)) if len(s) > 1 else np.nan,
        "min": float(np.min(s)),
        "p5": float(np.percentile(s, 5)),
        "p25": float(np.percentile(s, 25)),
        "median": float(np.percentile(s, 50)),
        "p75": float(np.percentile(s, 75)),
        "p95": float(np.percentile(s, 95)),
        "max": float(np.max(s)),
    }, name=name)

summary_df = pd.concat([
    summarize("Local distance d_local (km)", s_local),
    summarize("Long distance d_long (km)", s_long),
    summarize("Weight w (tonnes)", s_w),
    summarize("Share p (fraction 0-1)", s_p),
    summarize("EF_t (kg CO2/tonne-km)", s_EFt),
], axis=1).T.round(3)

print(summary_df)

plt.figure()
plt.hist(s_local, bins=50, density=True)
plt.title("Local-Distance Trip Distribution")
plt.xlabel("Travel Distance (kilometers)")
plt.ylabel("Probability Density")
plt.tight_layout()
plt.savefig("local_distance_distribution.pdf", bbox_inches="tight")

plt.figure()
plt.hist(s_long, bins=50, density=True)
plt.title("Long-Distance Trip Distribution")
plt.xlabel("Travel Distance (kilometers)")
plt.ylabel("Probability Density")
plt.tight_layout()
plt.savefig("long_distance_distribution.pdf", bbox_inches="tight")

plt.figure()
plt.hist(s_w, bins=50, density=True)
plt.title("Biomass Payload Weight Distribution")
plt.xlabel("Payload Weight (tonnes)")
plt.ylabel("Probability Density")
plt.tight_layout()
plt.savefig("payload_weight_distribution.pdf", bbox_inches="tight")

plt.title("Share p"); plt.xlabel("Share taking long route (fraction)"); plt.ylabel("Density")
plt.figure(); plt.hist(s_EFt, bins=50, density=True)

plt.figure()
plt.hist(s_EFt, bins=50, density=True)
plt.title("Transportation Emission Factor Distribution")
plt.xlabel("Emission Factor (kg CO2 / tonne-km)")
plt.ylabel("Probability Density")
plt.tight_layout()
plt.savefig("emission_factor_distribution.pdf", bbox_inches="tight")

plt.show()


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = "/Users/a20560/Desktop/backgroundss/useful/Data from Toronto.xlsx"

df_long = pd.read_excel(file_path, sheet_name="Dlong",  header=1)
df_short = pd.read_excel(file_path, sheet_name="Dshort", header=0)

s_dlong = df_long["Distance Long (dlong)"].dropna().astype(float)            
s_dshort = df_short["Total Distance(km)"].dropna().astype(float)             
s_weight = (
    df_long["Weight (Assume Woodchip Density (160 kg/m3))"]
    .dropna().astype(float) / 1000.0                                         
)

n_sims = 10_000
rng = np.random.default_rng(42)

def sample_empirical(series, size, rng):
    """Bootstrap samples from an empirical distribution."""
    vals = series.to_numpy()
    idx = rng.integers(0, len(vals), size=size)
    return vals[idx]

def sample_lognormal_from_range(q_low, q_high, size, rng):
    """
    Approximate a lognormal so that q_low and q_high are
    roughly the 2.5% and 97.5% quantiles.
    """
    log_low = np.log(q_low)
    log_high = np.log(q_high)
    sigma = (log_high - log_low) / (2 * 1.96)
    mu = 0.5 * (log_low + log_high)
    return rng.lognormal(mean=mu, sigma=sigma, size=size)

def summarize(name, s):
    """Summary table in the style of your previous notebook."""
    s = pd.Series(s)
    return pd.Series(
        {
            "count": len(s),
            "mean": s.mean(),
            "std": s.std(ddof=1),
            "min": s.min(),
            "p5": np.percentile(s, 5),
            "p25": np.percentile(s, 25),
            "median": np.percentile(s, 50),
            "p75": np.percentile(s, 75),
            "p95": np.percentile(s, 95),
            "max": s.max(),
        },
        name=name,
    )

d_long_sim = sample_empirical(s_dlong,  n_sims, rng)   
d_short_sim = sample_empirical(s_dshort, n_sims, rng)  

w_sim = sample_empirical(s_weight, n_sims, rng)        

FE_long_sim = sample_lognormal_from_range(29.0, 47.0, n_sims, rng)

FE_short_sim = sample_lognormal_from_range(22.0, 32.0, n_sims, rng)

p_min, p_mode, p_max = 1.0, 1.3, 1.5
p_sim = rng.triangular(p_min, p_mode, p_max, size=n_sims)

EF_CO2 = 2.68 

M_total_sim = (
    (FE_long_sim / 100.0) * d_long_sim
    + (FE_short_sim / 100.0) * (p_sim * d_short_sim)
) * EF_CO2

M_total_series = pd.Series(M_total_sim, name="M_total (kg CO2)")


summary_df = pd.concat(
    [
        summarize("d_long (km)", d_long_sim),
        summarize("d_short (km)", d_short_sim),
        summarize("Weight w (tonnes)", w_sim),
        summarize("FE_long (L/100 km)", FE_long_sim),
        summarize("FE_short (L/100 km)", FE_short_sim),
        summarize("PVI factor p", p_sim),
        summarize("M_total (kg CO2)", M_total_series),
    ],
    axis=1,
).T.round(3)

print(summary_df)

plt.figure()
plt.hist(M_total_sim, bins=50, density=True)
plt.title("Per-Trip Transportation Emission Distribution")
plt.xlabel("Total Per-Trip Emissions (kilograms CO2)")
plt.ylabel("Probability Density")
plt.tight_layout()
plt.savefig("M_total_histogram.pdf", bbox_inches="tight")
plt.show()


# In[10]:


total_biomass = 8122.08 
truck_payload = 25      
N_trips = total_biomass / truck_payload
print(N_trips)


# In[11]:


E_system_sim = M_total_sim * N_trips


# In[12]:


summary = {
    "mean": np.mean(E_system_sim),
    "p5": np.quantile(E_system_sim, 0.05),
    "p50": np.quantile(E_system_sim,0.5),
    "p95": np.quantile(E_system_sim,0.95)
}
summary


# In[13]:


import matplotlib.pyplot as plt
import numpy as np

E_system_tonnes = E_system_sim / 1000

plt.figure()
plt.hist(E_system_tonnes, bins=40, density=True)
plt.axvline(np.mean(E_system_tonnes), linestyle='--', label='Mean')
plt.axvline(np.quantile(E_system_tonnes, 0.05), linestyle=':', label='P5')
plt.axvline(np.quantile(E_system_tonnes, 0.95), linestyle=':', label='P95')
plt.xlabel('Total System Emissions (tonnes CO2)')
plt.ylabel('Probability Density')
plt.title('System Level Total Transportation Emission Distribution')
plt.legend()
plt.savefig('system_emissions_distribution.pdf', bbox_inches='tight')
plt.show()


# In[9]:


import numpy as np
E_system = np.array(E_system_sim)
np.save("E_system_kgCO2.npy", E_system)
print("Saved system emissions:", E_system.shape)


# BAU vs. MULCH

# E_processing,biochar: 100 kg CO₂e / t biomass     
# -> E_processing,biochar=0.1tCO2/t biomass   
# Ref: Tadele_Debela_201909_msc 

# C_storage,biochar: −864 to −885 kg CO₂e per tonne biomass   
# 62–66% comes from biochar carbon sequestration   
# Cstorage≈−0.6 tCO2/t biomass   
# Ref: American Chemical Society

# In[22]:


biomass_total = 8122.08
E_processing_biochar = biomass_total * 100
C_storage_biochar = biomass_total * 600
E_biochar_sim = (
    E_system
    + E_processing_biochar
    - C_storage_biochar
)
print(E_biochar_sim)


# In[24]:


E_mulch_sim = E_system_sim
print(E_system_sim)


# In[25]:


delta_sim = E_biochar_sim - E_mulch_sim
print(delta_sim)

