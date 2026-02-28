# PrecastAI-Q Prototype (Hackathon MVP)

## 1) What exactly to build (realistic MVP)
Build a demo prototype that predicts concrete strength and optimizes curing strategy and de-mould time.

### Inputs
- Mix parameters (`cement`, `water_cement_ratio`, `admixture`)
- Environment (`temperature`, `humidity`)
- Curing type (`steam`, `water`, `ambient`)

### Outputs
- Predicted strength curve over time
- Optimal de-mould time
- Best curing strategy
- Estimated cost
- Cycle time comparison (baseline vs optimized)
- What-If scenario simulation with strategy recalculation:
   - temperature +5°C
   - steam curing unavailable
   - labour availability -20%
- ROI business case metrics:
   - annual savings
   - payback period
   - ROI %
- Production risk score with Green/Yellow/Red band
- ESG impact estimates:
   - annual energy savings
   - annual CO₂ reduction (tons)
- Phased implementation strategy (Pilot → Multi-yard → ERP)

All outputs are simulated using `data/sample_production_data.csv`.

## 2) Folder structure

```text
precast_ai_q/
│
├── data/
│   └── sample_production_data.csv
├── strength_model.py
├── qpso_optimizer.py
├── fitness.py
├── app.py
├── requirements.txt
└── README.md
```

## 3) Step-by-step implementation

### Step 1: Dataset
- Use `data/sample_production_data.csv`.
- Columns: `cement, water_cement_ratio, admixture, temperature, humidity, curing_type, time_hours, strength`.

### Step 2: Strength prediction model
- `strength_model.py` trains a `RandomForestRegressor`.
- It one-hot encodes `curing_type`.
- Exposes training and prediction helpers.

### Step 3: QPSO optimizer
- `qpso_optimizer.py` contains a bounded custom QPSO implementation.
- Optimizes six continuous parameters: mix + environment + time.

### Step 4: Fitness function
- `fitness.py` computes weighted objective:
  - cycle time
  - estimated cost
  - energy proxy
- Applies strong penalty when predicted strength is below required threshold.

### Step 5: Dashboard integration
- `app.py` loads model, runs QPSO for each curing type, and picks best candidate.
- Visualizes:
  - metrics (strategy, de-mould time, strength, cost)
  - cycle/cost comparisons
  - baseline vs optimized strength curves
   - interactive ROI calculator
   - production risk score indicator
   - ESG sustainability estimator
   - phased implementation roadmap

## 4) Starter Python code for key parts

- Strength prediction starter: see `strength_model.py`
- QPSO starter: see `qpso_optimizer.py`
- Fitness function starter: see `fitness.py`

## 5) How to connect everything

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. In the UI:
   - Set baseline input values
   - Set required early strength
   - Click **Run QPSO Optimization**
4. Review generated outputs and compare baseline vs optimized strategy.
5. Enable **What-If Simulation Mode** to stress-test strategy under field constraints.
6. Use **ROI Calculator** for financial impact in ₹.
7. Use **ESG / Sustainability Mode** for annual CO₂ reduction estimates.

## Demo checklist
- Show sample dataset
- Show model quality metrics
- Run optimization live
- Show candidate table and selected best curing strategy
- Explain cycle time and cost impact
- Show ROI numbers and payback timeline
- Show production risk score color shift (baseline vs optimized)
- Show annual CO₂ reduction in ESG mode
- Close with phased implementation plan
