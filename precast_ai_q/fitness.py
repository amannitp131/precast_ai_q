from strength_model import predict_strength


def estimate_cost(cement, admixture, curing_type, temperature):
    material_cost = cement * 0.12 + admixture * 4.5
    curing_cost_map = {
        "steam": 7.0,
        "water": 3.5,
        "ambient": 1.5,
    }
    energy_cost = max(0.0, temperature - 20.0) * 0.06

    return float(material_cost + curing_cost_map.get(curing_type, 2.0) + energy_cost)


def fitness_function(
    params,
    strength_bundle,
    curing_type,
    required_strength=20.0,
    temperature_delta=0.0,
    labour_availability=1.0,
    predictor=None,
):
    cement, wc_ratio, admixture, temp, humidity, time_hours = params
    effective_temp = temp + temperature_delta
    safe_labour = max(0.5, min(1.2, labour_availability))

    if predictor is None:
        predicted_strength = predict_strength(
            bundle=strength_bundle,
            cement=cement,
            water_cement_ratio=wc_ratio,
            admixture=admixture,
            temperature=effective_temp,
            humidity=humidity,
            curing_type=curing_type,
            time_hours=time_hours,
        )
    else:
        predicted_strength = predictor(
            cement=cement,
            water_cement_ratio=wc_ratio,
            admixture=admixture,
            temperature=effective_temp,
            humidity=humidity,
            curing_type=curing_type,
            time_hours=time_hours,
        )

    if predicted_strength < required_strength:
        shortfall = required_strength - predicted_strength
        return float(9999 + shortfall * 100)

    cycle_time = max(1.0, time_hours / safe_labour)
    cost = estimate_cost(cement, admixture, curing_type, effective_temp)
    energy = effective_temp * 0.05
    labour_penalty = (1.0 - safe_labour) * 10.0

    return float(0.5 * cycle_time + 0.3 * cost + 0.2 * energy + labour_penalty)
