from __future__ import annotations

import numpy as np


def nelson_emc(temp_c, rh):
    """Nelson equilibrium moisture content, bounded to plausible percent."""
    rh = np.asarray(rh, dtype=float)
    temp_c = np.asarray(temp_c, dtype=float)
    emc = np.where(
        rh <= 10,
        0.03 + 0.2626 * rh - 0.00104 * rh * temp_c,
        np.where(rh <= 50, 2.22 - 0.160 * rh + 0.01660 * temp_c,
                 21.06 - 0.4944 * rh + 0.005565 * rh ** 2 - 0.00063 * rh * temp_c),
    )
    return np.clip(emc, 1.0, 40.0)


def evolve_fm(initial_fm, temp_c, rh, drying_tau_hours=10.0, wetting_tau_hours=6.0):
    """Advance a moisture state hourly toward EMC with asymmetric time constants."""
    temp_c = np.asarray(temp_c, dtype=float)
    rh = np.asarray(rh, dtype=float)
    output = np.empty_like(temp_c, dtype=float)
    state = np.asarray(initial_fm, dtype=float)
    for index in range(len(output)):
        equilibrium = nelson_emc(temp_c[index], rh[index])
        tau = np.where(equilibrium < state, drying_tau_hours, wetting_tau_hours)
        state = state + (equilibrium - state) * (1.0 - np.exp(-1.0 / tau))
        output[index] = state
    return output
