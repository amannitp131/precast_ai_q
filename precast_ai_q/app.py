import numpy as np
import pandas as pd
import streamlit as st

from fitness import estimate_cost, fitness_function
from qpso_optimizer import QPSO
from strength_model import predict_strength, predict_strength_curve, train_strength_model

st.set_page_config(page_title="PrecastAI-Q Prototype", layout="wide")
st.title("PrecastAI-Q Prototype")
st.caption("AI strength prediction + QPSO optimization for demo-ready precast strategy simulation")


@st.cache_resource
def load_model():
    return train_strength_model()


strength_bundle = load_model()
curing_types = strength_bundle["curing_types"]


def energy_kwh_per_element(curing_type, temperature):
    base_map = {
        "steam": 28.0,
        "water": 16.0,
        "ambient": 8.0,
    }
    base = base_map.get(curing_type, 12.0)
    thermal_load = max(0.0, temperature - 20.0) * 0.55
    return float(base + thermal_load)


def production_risk_score(predicted_strength, required_strength, temperature, humidity, curing_type):
    margin = predicted_strength - required_strength
    if margin >= 8:
        strength_risk = 5.0
    elif margin >= 5:
        strength_risk = 15.0
    elif margin >= 3:
        strength_risk = 35.0
    elif margin >= 1:
        strength_risk = 60.0
    else:
        strength_risk = 90.0

    climate_risk = min(100.0, abs(temperature - 30.0) * 3.0 + abs(humidity - 60.0) * 1.2)
    energy_risk = min(100.0, ((energy_kwh_per_element(curing_type, temperature) - 8.0) / 25.0) * 100.0)

    total_risk = 0.5 * strength_risk + 0.3 * climate_risk + 0.2 * energy_risk
    return float(max(0.0, min(100.0, total_risk)))


def risk_band(score):
    if score <= 35:
        return "Green", "🟢"
    if score <= 65:
        return "Yellow", "🟡"
    return "Red", "🔴"

with st.expander("Model Training Info", expanded=False):
    st.write(f"Train score (R²): {strength_bundle['train_score']:.3f}")
    st.write(f"Test score (R²): {strength_bundle['test_score']:.3f}")

st.subheader("Baseline Input")
col1, col2, col3 = st.columns(3)

with col1:
    base_cement = st.number_input("Cement (kg/m³)", min_value=250.0, max_value=600.0, value=400.0, step=1.0)
    base_wc = st.number_input("Water-cement ratio", min_value=0.25, max_value=0.80, value=0.45, step=0.01)

with col2:
    base_admixture = st.number_input("Admixture (%)", min_value=0.0, max_value=4.0, value=1.2, step=0.1)
    base_temp = st.number_input("Temperature (°C)", min_value=10.0, max_value=90.0, value=30.0, step=1.0)

with col3:
    base_humidity = st.number_input("Humidity (%)", min_value=20.0, max_value=100.0, value=60.0, step=1.0)
    base_time = st.number_input("Target de-mould time (hours)", min_value=2.0, max_value=48.0, value=8.0, step=1.0)

baseline_curing = st.selectbox("Baseline curing type", options=curing_types, index=0)
required_strength = st.number_input("Required early strength (MPa)", min_value=10.0, max_value=50.0, value=20.0, step=1.0)

st.subheader("Optimization Settings")
opt_col1, opt_col2 = st.columns(2)
with opt_col1:
    particles = st.slider("Particles", min_value=10, max_value=80, value=30, step=5)
with opt_col2:
    iterations = st.slider("Iterations", min_value=10, max_value=120, value=50, step=5)

st.subheader("What-If Simulation Mode")
enable_what_if = st.toggle("Enable scenario simulation", value=False)
temp_increase_by_5 = False
steam_unavailable = False
labour_reduced_20 = False

if enable_what_if:
    w1, w2, w3 = st.columns(3)
    with w1:
        temp_increase_by_5 = st.checkbox("What if temperature increases by 5°C?", value=False)
    with w2:
        steam_unavailable = st.checkbox("What if steam curing unavailable?", value=False)
    with w3:
        labour_reduced_20 = st.checkbox("What if labour reduced by 20%?", value=False)

temperature_delta = 5.0 if temp_increase_by_5 else 0.0
labour_availability = 0.8 if labour_reduced_20 else 1.0

available_curing_types = [c for c in curing_types if not (steam_unavailable and c == "steam")]

if enable_what_if:
    active_flags = []
    if temp_increase_by_5:
        active_flags.append("Temp +5°C")
    if steam_unavailable:
        active_flags.append("Steam unavailable")
    if labour_reduced_20:
        active_flags.append("Labour -20%")

    if active_flags:
        st.info(f"Active scenario: {', '.join(active_flags)}")
    else:
        st.info("What-If mode enabled with no active scenario changes.")

if st.button("Run QPSO Optimization", type="primary"):
    if not available_curing_types:
        st.error("No curing strategy available under this scenario.")
        st.stop()

    bounds = [
        (300.0, 500.0),
        (0.30, 0.60),
        (0.50, 2.50),
        (20.0, 70.0),
        (40.0, 95.0),
        (4.0, 24.0),
    ]

    candidate_rows = []

    for curing in available_curing_types:
        def objective(x, curing_type=curing):
            return fitness_function(
                x,
                strength_bundle,
                curing_type,
                required_strength,
                temperature_delta=temperature_delta,
                labour_availability=labour_availability,
            )

        optimizer = QPSO(
            fitness_function=objective,
            dim=6,
            bounds=bounds,
            particles=particles,
            iterations=iterations,
            seed=42,
        )

        solution, score = optimizer.optimize()
        predicted = predict_strength(
            bundle=strength_bundle,
            cement=solution[0],
            water_cement_ratio=solution[1],
            admixture=solution[2],
            temperature=min(90.0, solution[3] + temperature_delta),
            humidity=solution[4],
            curing_type=curing,
            time_hours=solution[5],
        )

        candidate_rows.append(
            {
                "curing_type": curing,
                "score": float(score),
                "cement": float(solution[0]),
                "water_cement_ratio": float(solution[1]),
                "admixture": float(solution[2]),
                "temperature": float(solution[3]),
                "effective_temperature": float(min(90.0, solution[3] + temperature_delta)),
                "humidity": float(solution[4]),
                "de_mould_time": float(solution[5]),
                "effective_cycle_time": float(solution[5] / labour_availability),
                "predicted_strength": float(predicted),
            }
        )

    candidates = pd.DataFrame(candidate_rows).sort_values("score").reset_index(drop=True)
    best = candidates.iloc[0]

    baseline_curing_effective = baseline_curing if baseline_curing in available_curing_types else available_curing_types[0]
    if baseline_curing != baseline_curing_effective:
        st.warning(
            f"Baseline curing '{baseline_curing}' is unavailable in this scenario. "
            f"Using '{baseline_curing_effective}' for baseline comparison."
        )

    baseline_strength = predict_strength(
        bundle=strength_bundle,
        cement=base_cement,
        water_cement_ratio=base_wc,
        admixture=base_admixture,
        temperature=min(90.0, base_temp + temperature_delta),
        humidity=base_humidity,
        curing_type=baseline_curing_effective,
        time_hours=base_time,
    )

    baseline_cost = estimate_cost(
        base_cement,
        base_admixture,
        baseline_curing_effective,
        min(90.0, base_temp + temperature_delta),
    )
    optimized_cost = estimate_cost(best["cement"], best["admixture"], best["curing_type"], best["effective_temperature"])

    baseline_cycle = base_time / labour_availability
    optimized_cycle = best["effective_cycle_time"]
    cycle_delta_pct = ((baseline_cycle - optimized_cycle) / baseline_cycle) * 100 if baseline_cycle else 0.0

    baseline_risk = production_risk_score(
        predicted_strength=baseline_strength,
        required_strength=required_strength,
        temperature=min(90.0, base_temp + temperature_delta),
        humidity=base_humidity,
        curing_type=baseline_curing_effective,
    )
    optimized_risk = production_risk_score(
        predicted_strength=best["predicted_strength"],
        required_strength=required_strength,
        temperature=best["effective_temperature"],
        humidity=best["humidity"],
        curing_type=best["curing_type"],
    )

    st.subheader("Optimization Result")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best curing strategy", str(best["curing_type"]).title())
    m2.metric("Optimal de-mould time", f"{best['effective_cycle_time']:.2f} h")
    m3.metric("Predicted strength at de-mould", f"{best['predicted_strength']:.2f} MPa")
    m4.metric("Estimated optimized cost", f"${optimized_cost:.2f}")

    st.subheader("Production Risk Score")
    r1, r2 = st.columns(2)
    with r1:
        baseline_band, baseline_icon = risk_band(baseline_risk)
        st.metric("Baseline risk score", f"{baseline_risk:.1f}/100")
        st.write(f"{baseline_icon} {baseline_band}")
    with r2:
        optimized_band, optimized_icon = risk_band(optimized_risk)
        st.metric("Optimized risk score", f"{optimized_risk:.1f}/100", delta=f"{- (baseline_risk - optimized_risk):.1f}")
        st.write(f"{optimized_icon} {optimized_band}")

    c1, c2 = st.columns(2)
    with c1:
        st.write("Baseline vs Optimized Cycle Time")
        comparison = pd.DataFrame(
            {
                "Scenario": ["Baseline", "Optimized"],
                "CycleTimeHours": [baseline_cycle, optimized_cycle],
            }
        )
        st.bar_chart(comparison.set_index("Scenario"))
        st.write(f"Cycle time improvement: {cycle_delta_pct:.1f}%")

    with c2:
        st.write("Baseline vs Optimized Cost")
        cost_df = pd.DataFrame(
            {
                "Scenario": ["Baseline", "Optimized"],
                "EstimatedCost": [baseline_cost, optimized_cost],
            }
        )
        st.bar_chart(cost_df.set_index("Scenario"))

    st.subheader("Predicted Strength Curve")
    time_points = np.arange(4, 25, 1)

    baseline_curve = predict_strength_curve(
        bundle=strength_bundle,
        fixed_params={
            "cement": base_cement,
            "water_cement_ratio": base_wc,
            "admixture": base_admixture,
            "temperature": min(90.0, base_temp + temperature_delta),
            "humidity": base_humidity,
        },
        curing_type=baseline_curing_effective,
        time_points=time_points,
    )

    optimized_curve = predict_strength_curve(
        bundle=strength_bundle,
        fixed_params={
            "cement": best["cement"],
            "water_cement_ratio": best["water_cement_ratio"],
            "admixture": best["admixture"],
            "temperature": best["effective_temperature"],
            "humidity": best["humidity"],
        },
        curing_type=best["curing_type"],
        time_points=time_points,
    )

    curve_df = pd.DataFrame(
        {
            "time_hours": time_points,
            "baseline_strength": baseline_curve,
            "optimized_strength": optimized_curve,
        }
    ).set_index("time_hours")

    st.line_chart(curve_df)

    st.subheader("All Curing Strategy Candidates")
    st.dataframe(candidates, use_container_width=True)

    st.session_state["latest_optimization"] = {
        "baseline_cycle": float(baseline_cycle),
        "optimized_cycle": float(optimized_cycle),
        "baseline_cost": float(baseline_cost),
        "optimized_cost": float(optimized_cost),
        "baseline_strength": float(baseline_strength),
        "optimized_strength": float(best["predicted_strength"]),
        "baseline_curing": str(baseline_curing_effective),
        "optimized_curing": str(best["curing_type"]),
        "baseline_temp": float(min(90.0, base_temp + temperature_delta)),
        "optimized_temp": float(best["effective_temperature"]),
        "baseline_humidity": float(base_humidity),
        "optimized_humidity": float(best["humidity"]),
        "required_strength": float(required_strength),
        "baseline_risk": float(baseline_risk),
        "optimized_risk": float(optimized_risk),
    }

    if enable_what_if:
        st.success("What-If simulation complete. Best strategy recalculated for the selected scenario.")
    else:
        st.success("Optimization complete. You can now compare baseline and optimized production strategy.")


st.subheader("ROI Calculator")
latest = st.session_state.get("latest_optimization", {})

roi_col1, roi_col2, roi_col3 = st.columns(3)
with roi_col1:
    elements_per_month = st.number_input("Elements per month", min_value=100, max_value=100000, value=2000, step=100)
    current_cycle_time = st.number_input(
        "Current cycle time (hours)",
        min_value=1.0,
        max_value=72.0,
        value=float(latest.get("baseline_cycle", base_time)),
        step=0.5,
    )

with roi_col2:
    current_cost = st.number_input(
        "Current cost per element (₹)",
        min_value=100.0,
        max_value=200000.0,
        value=2500.0,
        step=100.0,
    )
    projected_cycle_time = st.number_input(
        "Projected optimized cycle time (hours)",
        min_value=1.0,
        max_value=72.0,
        value=float(latest.get("optimized_cycle", max(1.0, base_time * 0.85))),
        step=0.5,
    )

with roi_col3:
    projected_cost = st.number_input(
        "Projected optimized cost per element (₹)",
        min_value=100.0,
        max_value=200000.0,
        value=float(max(100.0, latest.get("optimized_cost", current_cost * 0.9))),
        step=100.0,
    )
    implementation_cost = st.number_input(
        "Implementation investment (₹)",
        min_value=50000.0,
        max_value=200000000.0,
        value=2500000.0,
        step=50000.0,
    )

annual_elements = elements_per_month * 12
annual_cost_savings = max(0.0, (current_cost - projected_cost) * annual_elements)
cycle_gain_ratio = max(0.0, (current_cycle_time - projected_cycle_time) / max(1.0, current_cycle_time))
annual_cycle_savings = cycle_gain_ratio * annual_elements * current_cost * 0.35
annual_savings = annual_cost_savings + annual_cycle_savings

monthly_savings = annual_savings / 12.0
if monthly_savings > 0:
    payback_months = implementation_cost / monthly_savings
else:
    payback_months = float("inf")

roi_pct = ((annual_savings - implementation_cost) / implementation_cost) * 100.0 if implementation_cost else 0.0

rm1, rm2, rm3 = st.columns(3)
rm1.metric("Annual savings", f"₹{annual_savings:,.0f}")
rm2.metric("Payback period", "N/A" if not np.isfinite(payback_months) else f"{payback_months:.1f} months")
rm3.metric("ROI % (Year 1)", f"{roi_pct:.1f}%")


st.subheader("ESG / Sustainability Mode")
enable_esg = st.toggle("Enable ESG impact estimator", value=True)
if enable_esg:
    esg_col1, esg_col2, esg_col3 = st.columns(3)
    with esg_col1:
        annual_elements_for_esg = st.number_input(
            "Annual elements for ESG estimate",
            min_value=1000,
            max_value=2000000,
            value=int(annual_elements),
            step=1000,
        )
        emission_factor = st.number_input(
            "Grid emission factor (kg CO₂ / kWh)",
            min_value=0.1,
            max_value=1.5,
            value=0.82,
            step=0.01,
        )

    with esg_col2:
        esg_baseline_curing = st.selectbox(
            "Baseline curing for ESG",
            options=curing_types,
            index=curing_types.index(latest.get("baseline_curing", baseline_curing))
            if latest.get("baseline_curing", baseline_curing) in curing_types
            else 0,
        )
        esg_baseline_temp = st.number_input(
            "Baseline effective temperature (°C)",
            min_value=10.0,
            max_value=90.0,
            value=float(latest.get("baseline_temp", base_temp)),
            step=1.0,
        )

    with esg_col3:
        esg_optimized_curing = st.selectbox(
            "Optimized curing for ESG",
            options=curing_types,
            index=curing_types.index(latest.get("optimized_curing", baseline_curing))
            if latest.get("optimized_curing", baseline_curing) in curing_types
            else 0,
        )
        esg_optimized_temp = st.number_input(
            "Optimized effective temperature (°C)",
            min_value=10.0,
            max_value=90.0,
            value=float(latest.get("optimized_temp", base_temp)),
            step=1.0,
        )

    baseline_energy_per_element = energy_kwh_per_element(esg_baseline_curing, esg_baseline_temp)
    optimized_energy_per_element = energy_kwh_per_element(esg_optimized_curing, esg_optimized_temp)
    annual_energy_saved = max(0.0, (baseline_energy_per_element - optimized_energy_per_element) * annual_elements_for_esg)
    annual_co2_reduction_tons = (annual_energy_saved * emission_factor) / 1000.0

    em1, em2, em3 = st.columns(3)
    em1.metric("Energy saved per element", f"{max(0.0, baseline_energy_per_element - optimized_energy_per_element):.2f} kWh")
    em2.metric("Annual energy saved", f"{annual_energy_saved:,.0f} kWh")
    em3.metric("Annual CO₂ reduction", f"{annual_co2_reduction_tons:,.2f} tons")

    if esg_baseline_curing == "steam" and esg_optimized_curing != "steam":
        st.success("Steam curing reduction detected: ESG impact improved.")


st.subheader("Phased Implementation Strategy")
p1, p2, p3 = st.columns(3)
with p1:
    st.markdown("**Phase 1 → Pilot Yard (0-3 months)**")
    st.markdown("- Deploy dashboard in one yard")
    st.markdown("- Validate strength prediction and cycle KPIs")
    st.markdown("- Train supervisors and QC team")
with p2:
    st.markdown("**Phase 2 → Multi-Yard Rollout (3-9 months)**")
    st.markdown("- Replicate model to additional yards")
    st.markdown("- Benchmark ROI, risk, and ESG across locations")
    st.markdown("- Standardize best curing playbooks")
with p3:
    st.markdown("**Phase 3 → ERP Integration (9-15 months)**")
    st.markdown("- Integrate with production planning/ERP")
    st.markdown("- Automate scheduling and procurement signals")
    st.markdown("- Enable executive-level portfolio analytics")
