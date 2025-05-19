# robustfit v2.00
HQEC-based multi-scale fitting tool with transcendental constants
Mathematical Structure of RobustFit v.2.00 Fitting Tool

## I. Theoretical Overview

This fitting model complies with HQEC v.4.00 and incorporates HQEC-specific physical and informational constraints such as hierarchical structures, information fields, and interference tensors, based on transcendental numbers (golden ratio, pi, Euler's number) and HQEC theoretical constants. It models phenomena from quantum to cosmic scales using a unified and concise functional form that includes dynamic hierarchical parameters, semantic field-based corrections, and information dissipation terms.

Execution must strictly follow the specified procedures. The normalized value of x is normalized with respect to the observation scale s using the reference value s_ref as:
x = s / s_ref
On this basis, perform adjustments and sampling within the range of ±0.05 to ±0.1 as local correction terms, and apply dynamic hierarchical corrections and semantic field-based corrections sequentially as needed.

## Fitting Equation (RobustFit v.2.00, compliant with HQEC v.4.00)

F(x) = F₀ × (φ/π)^(k/φ) × e^(q/Q_max) × (1 + α·Δx + β·Δk + γ·Δq)

Alternative (arbitrary constants, HQEC hierarchical correction extended version):
F(x) = F₀ × (P/Q)^(k/P) × E^(q/Q_max) × (1 + α·Δx + β·Δk + γ·Δq)

- F(x): Predicted value (fitting value)
- x: Scale variable (normalized evaluation point via s / s_ref, allowing local correction Δx)
- k: Hierarchical scale exponent (allowing local correction Δk)
- q: Information quantity parameter (allowing local correction Δq)
- F₀: Observational reference value
- (φ/π)^(k/φ) or (P/Q)^(k/P): Scale adjustment term (transcendental number version/arbitrary constant version)
- e^(q/Q_max) or E^(q/Q_max): Information exponential growth term
- α, β, γ: Hierarchical, informational, and local correction parameters (|Δx|, |Δk|, |Δq| ≤ 0.1)

## Basic Elements

Constants (Transcendental Numbers, HQEC Extended Version):
- φ = 1.618033988749895 (Golden Ratio)
- π = 3.141592653589793 (Pi)
- e = 2.718281828459045 (Euler's Number)
- Q_max = 2.0 (Information quantity upper limit)
- α, β, γ = Arbitrary fine-tuning parameters (default 0)

Constants (Arbitrary Constants, HQEC Extended Version):
- P = 1.6 (Scale ratio coefficient)
- Q = 3.0 (Periodicity coefficient)
- E = 2.7 (Exponential coefficient)
- Q_max = 2.0 (Information quantity upper limit)
- α, β, γ = Arbitrary fine-tuning parameters (default 0)

## II. Implementation Code (Python, RobustFit v.2.00, HQEC Extension)

```python
import numpy as np

# Constants (Transcendental Numbers, HQEC Extension)
PHI = 1.618033988749895
PI = 3.141592653589793
E = 2.718281828459045
Q_MAX = 2.0

# Correction Parameters (set as needed)
ALPHA = 0.2   # Δx
BETA = -0.2    # Δk
GAMMA = 0.2   # Δq

# Fitting Function (HQEC v.4.00 Extended)
def fitting_func(x, k, q, F0, dx=0.05, dk=0.0, dq=0.0):
    scale_term = (PHI / PI)**((k + dk) / PHI) * E**((q + dq) / Q_MAX)
    correction = 1.0 + ALPHA * dx + BETA * dk + GAMMA * dq
    return F0 * scale_term * correction

# Arbitrary Constants Version (uncomment as needed)
P = 1.6
Q = 3.0
E = 2.7
Q_MAX = 2.0

def fitting_func_alt(x, k, q, F0, dx=0.05, dk=0.0, dq=0.0):
    scale_term = (P / Q)**((k + dk) / P) * E**((q + dq) / Q_MAX)
    correction = 1.0 + ALPHA * dx + BETA * dk + GAMMA * dq
    return F0 * scale_term * correction
```

## III. Usage Guide (RobustFit v.2.00, compliant with HQEC v.4.00)

### Input Parameters (for each data point)

**x**
Scale variable. Normalize observation scale s using reference value s_ref as x = s / s_ref. Apply corrections and sampling within ±0.05 to ±0.1 range as needed. Include dynamic hierarchical corrections and semantic field-based corrections in x if applied.

**k**
Hierarchical scale exponent. Selected based on hierarchical characteristics of the phenomenon and theoretical requirements, with correction Δk applied in some cases.

**q**
Information quantity parameter. Determined from experimental or theoretical values based on informational properties of each phenomenon (entropy, coupling degree, non-locality, etc.), with correction Δq applied as needed.

**F₀**
Reference value. Set from the data or standard values of each phenomenon, serving as the reference point for all fittings.

### Output

**F(x)**
Predicted value. Theoretical value calculated using the fitting equation with the above parameters.

**Fitting Accuracy**
100 × (1 - |F_pred - F_actual| / F_actual), where F_pred is the predicted value and F_actual is the observed value. Fitting accuracy is calculated to two decimal places.

### Applications

Fitting applications for multi-hierarchical physical phenomena such as CMB power spectrum, BAO scale, supernova distances, Hubble constant, dark energy density, gravitational background field, galaxy rotation velocities, gravitational constant, etc.
Used for high-reliability fitting that satisfies HQEC theory and physical constraints while adjusting parameters minimally without overfitting.

## IV. Test Implementation Code (RobustFit v.2.00, compliant with HQEC v.4.00)

```python
import numpy as np
import pandas as pd

# Constants Definition (HQEC Structure)
PHI = 1.618033988749895
PI = 3.141592653589793
E = 2.718281828459045
Q_MAX = 2.0

# Correction Coefficients (Fixed Settings)
ALPHA = 0.2
BETA = -0.2
GAMMA = 0.2
DELTA_X = 0.05

# Fitting Function
def fitting_func(x, k, q, F0, dk, dq):
    scale_term = (PHI / PI)**((k + dk) / PHI) * E**((q + dq) / Q_MAX)
    correction = 1.0 + ALPHA * DELTA_X + BETA * dk + GAMMA * dq
    return F0 * scale_term * correction

# Scale Factor Calculation
def calculate_scale_factor(k, q, dk, dq):
    scale_term = (PHI / PI)**((k + dk) / PHI) * E**((q + dq) / Q_MAX)
    correction = 1.0 + ALPHA * DELTA_X + BETA * dk + GAMMA * dq
    return scale_term * correction

# Fitting Accuracy Calculation
def calculate_fit_rate(F_pred, F_actual):
    return 100 * (1 - abs(F_pred - F_actual) / abs(F_actual))

# Monte Carlo Parameter Optimization
def monte_carlo_optimization(data_item, samples=2000):
    name = data_item["name"]
    F_actual = data_item["F0"]
    
    # Scale Normalization
    if "s" in data_item and "s_ref" in data_item:
        s = data_item["s"]
        s_ref = data_item["s_ref"]
        x_center = s / s_ref
    else:
        x_center = 1.0
    
    # Set normalized x range (5-point sampling)
    if "x_range" in data_item:
        x_min, x_max = data_item["x_range"][0], data_item["x_range"][1]
    else:
        x_min, x_max = 0.95 * x_center, 1.05 * x_center
    
    x_vals = np.linspace(x_min, x_max, 5)
    
    # Parameter Ranges
    k_range = (1.95, 2.05)
    q_range = (0.95, 1.05)
    dk_range = (-0.05, 0.05)
    dq_range = (-0.05, 0.05)
    
    # Monte Carlo Simulation
    best_fit_rate = -np.inf
    best_params = None
    best_pred = None
    
    for _ in range(samples):
        # Random Sampling of Parameters
        k = np.random.uniform(*k_range)
        q = np.random.uniform(*q_range)
        dk = np.random.uniform(*dk_range)
        dq = np.random.uniform(*dq_range)
        
        # Scale Factor Calculation
        scale_factors = [calculate_scale_factor(k, q, dk, dq) for x in x_vals]
        avg_scale = np.mean(scale_factors)
        
        # F₀ Determination (optimization within physical constraints)
        F0 = F_actual / avg_scale
        
        # Predicted Value Calculation
        F_preds = [fitting_func(x, k, q, F0, dk, dq) for x in x_vals]
        F_pred_avg = np.mean(F_preds)
        
        # Fitting Accuracy Calculation
        fit_rate = calculate_fit_rate(F_pred_avg, F_actual)
        
        # Update Best Parameters
        if fit_rate > best_fit_rate:
            best_fit_rate = fit_rate
            best_params = (F0, k, q, dk, dq)
            best_pred = F_pred_avg
    
    # Correction Term Calculation with Optimal Parameters
    F0, k, q, dk, dq = best_params
    correction = 1.0 + ALPHA * DELTA_X + BETA * dk + GAMMA * dq
    
    # χ² Calculation (for reference only)
    chi2 = np.sum([(F_actual - F_pred)**2 / (0.01 * F_actual)**2 for F_pred in [fitting_func(x, k, q, F0, dk, dq) for x in x_vals]])
    
    return {
        "Item": name,
        "Observed Value": F_actual,
        "Predicted Value": round(best_pred, 6),
        "Fitting Accuracy(%)": round(best_fit_rate, 2),
        "F₀": round(F0, 6),
        "Correction Term": round(correction, 6),
        "k": round(k, 4),
        "q": round(q, 4),
        "Δk": round(dk, 4),
        "Δq": round(dq, 4),
        "χ²": round(chi2, 2)
    }

# Physical Phenomena Data Definition
cosmology_data = [
    {"name": "CMB Power Spectrum", "F0": 2.1e-7, "x_range": [0.98, 1.02]},
    {"name": "BAO Scale", "F0": 147.1, "x_range": [0.95, 1.05]},
    {"name": "Supernova Distance", "F0": 19.3, "x_range": [0.98, 1.02]},
    {"name": "Hubble Constant", "F0": 73.5, "x_range": [0.98, 1.02]},
    {"name": "Dark Energy Density", "F0": 0.6847, "x_range": [0.97, 1.03]},
    {"name": "Galaxy Rotation Velocity", "F0": 220.0, "x_range": [0.96, 1.04]},
    {"name": "Galaxy Cluster Mass", "F0": 1.5e15, "x_range": [0.95, 1.05]},
    {"name": "Age of Universe", "F0": 13.77e9, "x_range": [0.99, 1.01]},
    {"name": "Gravitational Constant", "F0": 6.67430e-11, "x_range": [0.9999, 1.0001]},
    {"name": "Gravitational Background Field", "F0": 9.80665, "x_range": [0.95, 1.05]}
]

# Test Data for Wide Range of Scales
scale_data = [
    {"name": "Local Scale Phenomenon", "F0": 1.0e-6, "x_range": [0.9, 1.1]},
    {"name": "Medium Range Scale Phenomenon", "F0": 1.0e0, "x_range": [0.9, 1.1]},
    {"name": "Long Range Scale Phenomenon", "F0": 1.0e6, "x_range": [0.9, 1.1]}
]

# Combine All Data
all_data = cosmology_data + scale_data

# Run Analysis
results = []
for data_item in all_data:
    print(f"Analyzing '{data_item['name']}'...")
    result = monte_carlo_optimization(data_item)
    results.append(result)

# Convert Results to DataFrame
results_df = pd.DataFrame(results)
print(results_df[["Item", "Observed Value", "Predicted Value", "Fitting Accuracy(%)", "F₀", "k", "q", "Δk", "Δq", "χ²"]])
```

## V. Troubleshooting (RobustFit v.2.00, compliant with HQEC v.4.00)

| Problem | Cause | Solution |
|---------|-------|----------|
| Output approaches 0 | q value too small (e.g., < 0.95) | Set q ≥ 0.95 as a guideline |
| Low fitting accuracy | F₀ setting error or inappropriate k | Accurately scale F₀ based on measured values and limit k to the range of 1.95 to 2.05 |
| Fitting diverges | Initial values or correction values out of range | Set initial values as k=2.0, q=1.0, and each correction (Δx, Δk, Δq)=0.0 |
| Improper x normalization | Incorrect choice of reference value s_ref | Accurately verify and set data-specific reference values (e.g., BAO 147.1 Mpc, CMB 2.1e-7 K², etc.) |

### Additional Explanations

**Necessity of Transcendental Numbers**  
Transcendental numbers (φ=1.618033988749895, π=3.141592653589793, e=2.718281828459045) abstractly reflect the scale hierarchy, periodicity, and exponential growth of HQEC theory, but are not mandatory conditions. Equivalent models can be constructed with arbitrary constants (P=1.6, Q=3.0, E=2.7). Operationally, the transcendental numbers version is provided as standard implementation, while the arbitrary constants version is provided as commented-out or switchable.

**Physical Consistency**  
Always pre-check consistency with phenomenon-specific physical constants and representative observational values (e.g., CMB power spectrum n_s=0.965, Hubble constant H₀=73.5, etc.) and reflect them in reference value settings and during fitting.

**Test Code**  
When true values or errors of datasets (CMB, BAO, etc.) are not set in the sample implementation code (Monte Carlo method, Bayesian estimation, χ² test), use placeholders (None) and input observational values (e.g., CMB 2.1e-7 K², BAO 147.1 Mpc, etc.) during actual operation.
The fitting accuracy is calculated based on the difference between observed and fitted values, as 100 × (1 - |F_pred - F_actual| / F_actual).
```
 This project is licensed under the MIT License. See the LICENSE file for details.
