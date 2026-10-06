import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scrubber_model import (
    ScrubberSimulationV34,
    PACKING_DATA,
    CONCENTRATION_BASES,
)

st.set_page_config(
    page_title="Scrubber Design Simulator V3.6",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


@dataclass(frozen=True)
class Inputs:
    packing: str
    diameter_m: float
    packed_height_m: float
    liquid_kg_h: float
    gas_m3_h: float
    flow_reference: str
    temperature_C: float
    pressure_Pa: float
    concentration_basis: str
    inlet_concentration: float
    acn_mass_fraction: float
    vac_mass_fraction: float
    target_concentration: float


def make_sim(i: Inputs) -> ScrubberSimulationV34:
    return ScrubberSimulationV34(
        Dolgu_Tipi=i.packing,
        Kolon_Capi_D=i.diameter_m,
        Dolgu_Yuksekligi_Z=i.packed_height_m,
        Sivi_Debisi_L=i.liquid_kg_h,
        Gaz_Debisi_Q=i.gas_m3_h,
        Debi_Referansi=i.flow_reference,
        Sicaklik_T=i.temperature_C,
        P_operating=i.pressure_Pa,
        Giris_TOC_N=i.inlet_concentration,
        ACN_Kutle_Kesri=i.acn_mass_fraction,
        VAc_Kutle_Kesri=i.vac_mass_fraction,
        Konsantrasyon_Bazi=i.concentration_basis,
    )


def hydraulic_badge(percent: float) -> str:
    if percent >= 100:
        return "🔴 FLOODING PREDICTED"
    if percent >= 90:
        return "🟠 NEAR FLOODING"
    if percent >= 80:
        return "🟡 HIGH HYDRAULIC LOADING"
    if percent >= 60:
        return "🟢 NORMAL DESIGN REGION"
    if percent >= 40:
        return "🔵 LOW–MODERATE HYDRAULIC LOADING"
    return "🔵 UNDERLOADED / HIGH CAPACITY MARGIN"


def result_row(i: Inputs, result: dict, z_req):
    return {
        "Packing": i.packing,
        "Basis": i.concentration_basis,
        "Inlet": result["inlet_concentration"],
        "Outlet": result["outlet_concentration"],
        "Removal_percent": 100 * result["overall_removal"],
        "Required_height_m": z_req,
        "Flood_percent": result["flooding_percent"],
        "dP_mbar_m": result["dPwet_mbar_m"],
        "A_ACN": result["ACN"]["coefficients"]["absorption_factor"],
        "A_VAc": result["VAc"]["coefficients"]["absorption_factor"],
        "HTU_OG_ACN_m": result["ACN"]["coefficients"]["HTU_OG_m"],
        "HTU_OG_VAc_m": result["VAc"]["coefficients"]["HTU_OG_m"],
        "Packing_factor_ft-1": result["Fp_ft"],
        "Hydraulic_regime": result["hydraulic_regime"],
    }


@st.cache_data(show_spinner=False)
def run_cached(data: dict, profiles: bool = True):
    i = Inputs(**data)
    sim = make_sim(i)
    result = sim.simulate(return_profiles=profiles)
    z_req = sim.required_height(i.target_concentration, basis=i.concentration_basis)
    return result, z_req


@st.cache_data(show_spinner=False)
def packing_comparison_cached(data: dict, selected_packings: tuple):
    base = Inputs(**data)
    rows = []
    for packing in selected_packings:
        try:
            i = Inputs(**{**asdict(base), "packing": packing})
            sim = make_sim(i)
            r = sim.simulate(return_profiles=False)
            z_req = sim.required_height(i.target_concentration, basis=i.concentration_basis)
            rows.append(result_row(i, r, z_req))
        except Exception as exc:
            rows.append({"Packing": packing, "Error": str(exc)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def sweep_cached(data: dict, parameter: str, start: float, stop: float, points: int):
    base = Inputs(**data)
    values = np.linspace(start, stop, points)
    param_map = {
        "Liquid flow (kg/h)": "liquid_kg_h",
        "Gas flow (m³/h)": "gas_m3_h",
        "Packed height (m)": "packed_height_m",
        "Column diameter (m)": "diameter_m",
        "Temperature (°C)": "temperature_C",
    }
    field = param_map[parameter]
    rows = []
    for v in values:
        try:
            i = Inputs(**{**asdict(base), field: float(v)})
            sim = make_sim(i)
            r = sim.simulate(return_profiles=False)
            rows.append({
                "x": float(v),
                f"Outlet ({i.concentration_basis})": r["outlet_concentration"],
                "Removal (%)": 100*r["overall_removal"],
                "Flooding (%)": r["flooding_percent"],
                "ΔP (mbar/m)": r["dPwet_mbar_m"],
                "A_ACN": r["ACN"]["coefficients"]["absorption_factor"],
                "A_VAc": r["VAc"]["coefficients"]["absorption_factor"],
            })
        except Exception:
            rows.append({
                "x": float(v),
                f"Outlet ({base.concentration_basis})": np.nan,
                "Removal (%)": np.nan,
                "Flooding (%)": np.nan,
                "ΔP (mbar/m)": np.nan,
                "A_ACN": np.nan,
                "A_VAc": np.nan,
            })
    return pd.DataFrame(rows)


st.title("🧪 Scrubber Design Simulator — V3.6")
st.caption(
    "ACN + Vinyl Acetate absorption into water | ODE mass transfer + GPDC flooding | "
    "A, HTU/NTU, temperature-dependent water properties, concentration-basis conversion and live calculation walkthrough"
)

with st.sidebar:
    st.header("Model Inputs")
    st.caption("Values are recalculated when you press **Run simulation**.")

    with st.form("input_form"):
        st.subheader("Column & Packing")
        packing = st.selectbox(
            "Packing",
            list(PACKING_DATA.keys()),
            index=list(PACKING_DATA.keys()).index("25mm Metal Pall Ring"),
        )
        diameter = st.number_input("Column diameter, D (m)", min_value=0.05, value=0.50, step=0.05, format="%.3f")
        height = st.number_input("Packed height, Z (m)", min_value=0.05, value=1.40, step=0.10, format="%.3f")

        st.subheader("Operating Conditions")
        liquid = st.number_input("Water flow (kg/h)", min_value=1.0, value=2500.0, step=50.0)
        gas = st.number_input("Gas flow (m³/h)", min_value=0.1, value=117.53, step=5.0, format="%.2f")
        flow_ref = st.selectbox("Gas-flow reference", ["Aktual", "Normal"], index=0)
        temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, value=22.0, step=1.0)
        pressure_bar = st.number_input("Operating pressure (bar abs)", min_value=0.10, value=1.01325, step=0.05, format="%.5f")

        st.subheader("Feed & Target")
        basis = st.selectbox(
            "Concentration basis",
            list(CONCENTRATION_BASES),
            index=0,
            help="mgVOC/Nm³ = actual VOC mass; mgC/Nm³ = carbon-equivalent mass; ppmv = gas-volume/mole basis.",
        )
        inlet_default = 5000.0 if basis != "ppmv" else 2000.0
        target_default = 20.0
        inlet = st.number_input(f"Inlet concentration ({basis})", min_value=0.000001, value=float(inlet_default), step=10.0)
        acn_pct = st.slider("ACN fraction of VOC mixture (mass %)", min_value=0.0, max_value=100.0, value=93.0, step=0.5)
        vac_pct = 100.0 - acn_pct
        st.caption(f"VAc mass fraction is automatically set to **{vac_pct:.1f}%**.")
        target = st.number_input(f"Target outlet ({basis})", min_value=0.000001, value=float(target_default), step=1.0)

        submitted = st.form_submit_button("▶ Run simulation", use_container_width=True, type="primary")

    st.divider()
    st.info(
        "The selected concentration basis is now used by the calculation itself — not only for display. "
        "ACN/VAc composition is still entered as VOC **mass fraction**."
    )

inputs = Inputs(
    packing=packing,
    diameter_m=float(diameter),
    packed_height_m=float(height),
    liquid_kg_h=float(liquid),
    gas_m3_h=float(gas),
    flow_reference=flow_ref,
    temperature_C=float(temp),
    pressure_Pa=float(pressure_bar)*1e5,
    concentration_basis=basis,
    inlet_concentration=float(inlet),
    acn_mass_fraction=float(acn_pct)/100.0,
    vac_mass_fraction=float(vac_pct)/100.0,
    target_concentration=float(target),
)

if "active_inputs_v34" not in st.session_state:
    st.session_state.active_inputs_v34 = asdict(inputs)
if submitted:
    st.session_state.active_inputs_v34 = asdict(inputs)

active = Inputs(**st.session_state.active_inputs_v34)

try:
    with st.spinner("Running scrubber model..."):
        result, z_req = run_cached(asdict(active), profiles=True)
except Exception as exc:
    st.error("Simulation could not be completed.")
    st.exception(exc)
    st.stop()

unit = active.concentration_basis

# -----------------------------------------------------------------------------
# TOP SUMMARY
# -----------------------------------------------------------------------------
metric_cols = st.columns(5)
metric_cols[0].metric("Outlet concentration", f"{result['outlet_concentration']:.3f} {unit}")
metric_cols[1].metric("Overall removal", f"{100*result['overall_removal']:.2f} %")
metric_cols[2].metric("Required height", "Not reached" if z_req is None else f"{z_req:.3f} m")
metric_cols[3].metric("Flooding", f"{result['flooding_percent']:.1f} %")
metric_cols[4].metric("Pressure drop", f"{result['dPwet_mbar_m']:.3f} mbar/m")

if result["outlet_concentration"] <= active.target_concentration:
    st.success(f"Target achieved: outlet ≤ {active.target_concentration:g} {unit}")
else:
    st.warning(f"Target not achieved at the current packed height of {active.packed_height_m:g} m.")

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "Overview", "Mass Transfer + A/HTU/NTU", "Water Properties", "Hydraulics",
    "Profiles", "Packing Comparison", "Parameter Sweep", "Method & Assumptions"
])
(tab_overview, tab_mt, tab_water, tab_hyd, tab_profiles, tab_compare, tab_sweep, tab_method) = tabs

with tab_overview:
    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Performance summary")
        perf = pd.DataFrame([
            ["Total", result["inlet_concentration"], result["outlet_concentration"], 100*result["overall_removal"]],
            ["ACN", result["ACN_in"], result["ACN_out"], 100*result["ACN_removal"]],
            ["VAc", result["VAc_in"], result["VAc_out"], 100*result["VAc_removal"]],
        ], columns=["Component", f"Inlet ({unit})", f"Outlet ({unit})", "Removal (%)"])
        st.dataframe(perf, use_container_width=True, hide_index=True)

        st.subheader("Concentration-basis conversion")
        conversion = pd.DataFrame([
            ["mgVOC/Nm³", result["inlet_by_basis"]["mgVOC/Nm³"], result["outlet_by_basis"]["mgVOC/Nm³"]],
            ["mgC/Nm³", result["inlet_by_basis"]["mgC/Nm³"], result["outlet_by_basis"]["mgC/Nm³"]],
            ["ppmv", result["inlet_by_basis"]["ppmv"], result["outlet_by_basis"]["ppmv"]],
        ], columns=["Basis", "Inlet", "Outlet"])
        st.dataframe(conversion, use_container_width=True, hide_index=True)
        st.caption("Conversions use ACN/VAc molecular weights, carbon atom counts and the model's normal-condition definition (0 °C, 1 atm).")

    with right:
        st.subheader("Hydraulic status")
        st.markdown(f"### {hydraulic_badge(result['flooding_percent'])}")
        st.progress(min(max(result["flooding_percent"]/100.0, 0.0), 1.0))
        st.write(f"**Operating gas velocity:** {result['Ug']:.4f} m/s")
        st.write(f"**Predicted flood velocity:** {result['U_flood']:.4f} m/s")
        st.write(f"**Total packed-bed ΔP:** {result['Total_dP_mbar']:.3f} mbar")
        st.write(f"**Packing factor:** {result['Fp_ft']:.2f} ft⁻¹ ({result['Fp_basis']})")

        st.subheader("Engineering warnings")
        if result["warnings"]:
            for warning in result["warnings"]:
                st.warning(warning)
        else:
            st.success("No model warnings for the selected case.")

    export = pd.DataFrame([result_row(active, result, z_req)])
    st.download_button(
        "⬇ Download current result as CSV",
        export.to_csv(index=False).encode("utf-8-sig"),
        file_name="scrubber_result_v34.csv",
        mime="text/csv",
    )

with tab_mt:
    st.subheader("Mass-transfer coefficients and requested diagnostics")
    rows = []
    for comp in ("ACN", "VAc"):
        c = result[comp]["coefficients"]
        rows.append({
            "Component": comp,
            "A = L/(mG)": c["absorption_factor"],
            "HTU_OG (m)": c["HTU_OG_m"],
            "NTU_OG = Z/HTU": c["NTU_OG"],
            "a_t (m²/m³)": c["a_t"],
            "a_e (m²/m³)": c["a_e"],
            "Wetting (%)": 100*c["wetting_fraction"],
            "Re_L": c["Re_L"],
            "Re_G": c["Re_G"],
            "Sc_L": c["Sc_L"],
            "Sc_G": c["Sc_G"],
            "k_L": c["k_L"],
            "k_G": c["k_G"],
            "K_G": c["K_G"],
            "Henry (Pa·m³/mol)": c["H"],
            "m = y*/x": c["m_x"],
            "Gas-film resistance (%)": 100*c["gas_resistance_fraction"],
            "Liquid-film resistance (%)": 100*c["liquid_resistance_fraction"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    a1, a2 = st.columns(2)
    a1.metric("ACN absorption factor", f"{result['ACN']['coefficients']['absorption_factor']:.3f}")
    a2.metric("VAc absorption factor", f"{result['VAc']['coefficients']['absorption_factor']:.3f}")
    st.info(
        "A = L/(mG) is reported separately for ACN and VAc because their equilibrium slopes m differ. "
        "HTU_OG is computed consistently from the model equation G/(K_G a_e P); NTU_OG = Z/HTU_OG."
    )

with tab_water:
    st.subheader("Temperature-dependent water properties")
    wp = result["water_properties"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Density, ρL", f"{wp['rho']:.2f} kg/m³")
    c2.metric("Viscosity, μL", f"{wp['mu']*1e3:.4f} mPa·s")
    c3.metric("Surface tension, σL", f"{wp['sigma']*1e3:.3f} mN/m")
    st.write(
        "These properties are recalculated at the selected scrubber temperature and are used directly in "
        "Reynolds/Schmidt numbers, wetted area, kL, liquid holdup, pressure drop and GPDC capacity calculations."
    )

    temps = np.linspace(max(0.0, active.temperature_C-15), min(100.0, active.temperature_C+25), 30)
    from scrubber_model import water_properties
    prop_df = pd.DataFrame([
        {"Temperature (°C)": t, "Density (kg/m³)": water_properties(float(t))["rho"],
         "Viscosity (mPa·s)": water_properties(float(t))["mu"]*1e3,
         "Surface tension (mN/m)": water_properties(float(t))["sigma"]*1e3}
        for t in temps
    ])
    st.dataframe(prop_df, use_container_width=True, hide_index=True)

with tab_hyd:
    st.subheader("Hydraulic capacity")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ug", f"{result['Ug']:.4f} m/s")
    c2.metric("U flood", f"{result['U_flood']:.4f} m/s")
    c3.metric("F-oper", f"{result['F_oper']:.4f} Pa⁰·⁵")
    c4.metric("F-flood", f"{result['F_flood']:.4f} Pa⁰·⁵")

    hydraulic_df = pd.DataFrame({
        "Parameter": [
            "Flood fraction", "Hydraulic regime", "F_LV at flood", "GPDC CP at flood",
            "Packing factor", "Packing-factor basis", "Wet ΔP", "Dry ΔP",
            "Total ΔP", "Kister-Gill flood ΔP diagnostic", "Operating ΔP / flood ΔP"
        ],
        "Value": [
            f"{result['flooding_percent']:.2f} %", result["hydraulic_regime"],
            f"{result['F_LV_flood']:.5f}", f"{result['CP_flood']:.5f}",
            f"{result['Fp_ft']:.3f} ft⁻¹", result["Fp_basis"],
            f"{result['dPwet_mbar_m']:.5f} mbar/m", f"{result['dPdry_Pa_m']:.3f} Pa/m",
            f"{result['Total_dP_mbar']:.5f} mbar",
            f"{result['dP_flood_mbar_m']:.3f} mbar/m", f"{result['dP_ratio_to_flood']:.5f}",
        ]
    })
    st.dataframe(hydraulic_df, use_container_width=True, hide_index=True)
    st.info("%Flood = Ug,oper / Ug,flood × 100. Kister–Gill flood ΔP remains a diagnostic only.")

with tab_profiles:
    st.subheader(f"Gas-phase concentration profile — {unit}")
    profile_sim = make_sim(active)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for comp in ("ACN", "VAc"):
        z = np.asarray(result[comp]["z"])
        y = np.asarray(result[comp]["y_profile"])
        conc = np.array([profile_sim.y_to_component_concentration(val, comp, unit) for val in y])
        ax.plot(z, conc, marker="o", markersize=2.5, label=comp)
    ax.set_xlabel("Packed height coordinate, z (m)")
    ax.set_ylabel(f"Gas concentration ({unit})")
    ax.set_title("Gas-phase concentration profile")
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig, clear_figure=True)
    st.caption("Coordinate starts at the gas-inlet / liquid-outlet end of the counter-current packed section.")

with tab_compare:
    st.subheader("Compare packings at the same operating conditions")
    selected = st.multiselect(
        "Packings", list(PACKING_DATA.keys()),
        default=["25mm Metal Pall Ring", "38mm Metal Pall Ring", "IMTP #25 (Metal)"]
    )
    if selected:
        comp_df = packing_comparison_cached(asdict(active), tuple(selected))
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download packing comparison",
            comp_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="packing_comparison_v34.csv", mime="text/csv"
        )
    else:
        st.info("Select at least one packing.")

with tab_sweep:
    st.subheader("Sensitivity / parameter sweep")
    ca, cb, cc, cd = st.columns(4)
    sweep_parameter = ca.selectbox(
        "Parameter",
        ["Liquid flow (kg/h)", "Gas flow (m³/h)", "Packed height (m)", "Column diameter (m)", "Temperature (°C)"]
    )
    defaults = {
        "Liquid flow (kg/h)": (max(100.0, active.liquid_kg_h*0.4), active.liquid_kg_h*1.6),
        "Gas flow (m³/h)": (max(1.0, active.gas_m3_h*0.4), active.gas_m3_h*2.0),
        "Packed height (m)": (max(0.1, active.packed_height_m*0.4), active.packed_height_m*2.0),
        "Column diameter (m)": (max(0.1, active.diameter_m*0.6), active.diameter_m*1.5),
        "Temperature (°C)": (max(0.0, active.temperature_C-10.0), min(100.0, active.temperature_C+20.0)),
    }
    d_start, d_stop = defaults[sweep_parameter]
    start = cb.number_input("Start", value=float(d_start))
    stop = cc.number_input("Stop", value=float(d_stop))
    npts = cd.slider("Points", min_value=5, max_value=50, value=16)

    if stop <= start:
        st.error("Stop must be greater than Start.")
    else:
        sweep_df = sweep_cached(asdict(active), sweep_parameter, float(start), float(stop), int(npts))
        renamed = sweep_df.rename(columns={"x": sweep_parameter})
        st.dataframe(renamed, use_container_width=True, hide_index=True)
        outlet_col = f"Outlet ({unit})"
        st.line_chart(sweep_df.set_index("x")[[outlet_col]], x_label=sweep_parameter, y_label=outlet_col)
        st.line_chart(sweep_df.set_index("x")[["Flooding (%)", "Removal (%)"]], x_label=sweep_parameter, y_label="Percent")
        st.line_chart(sweep_df.set_index("x")[["A_ACN", "A_VAc"]], x_label=sweep_parameter, y_label="Absorption factor")
        st.download_button(
            "⬇ Download sweep data", renamed.to_csv(index=False).encode("utf-8-sig"),
            file_name="parameter_sweep_v34.csv", mime="text/csv"
        )

with tab_method:
    st.header("Methods & Assumptions — Full Calculation Guide")
    st.info(
        "This section documents the model **as it is actually implemented in the code**. "
        "It is intended to let a new user follow the calculation from raw inputs to outlet concentration, "
        "required packed height, pressure drop and flooding capacity. Equations labelled as screening or "
        "fallback methods should not be interpreted as vendor-rating equations."
    )

    method_sim = make_sim(active)
    area_col = method_sim.A
    q_actual = method_sim.Q
    nG = method_sim.gas_molar_flow()
    nL = method_sim.liquid_molar_flow()
    G_mass_flux = method_sim.gas_mass_flux()
    L_mass_flux = method_sim.liquid_mass_flux()
    Ug = method_sim.gas_superficial_velocity()

    st.subheader("Calculation roadmap")
    st.markdown(
        """
The model follows the calculation chain below:

**Inputs → unit/basis conversion → gas & water properties → superficial/molar fluxes → Henry equilibrium → Onda wetted area → Onda gas/liquid film coefficients → overall gas-side coefficient → counter-current ODE integration → outlet concentration → required height → pressure-drop screening → GPDC flooding capacity → diagnostics.**

ACN and vinyl acetate (VAc) are solved as two independent dilute solutes and are recombined only when total outlet concentration is reported.
        """
    )

    # -------------------------------------------------------------------------
    # LIVE WORKED EXAMPLE — ALL NUMBERS COME FROM THE ACTIVE SIMULATION CASE
    # -------------------------------------------------------------------------
    st.subheader("Live Worked Example — current inputs, current calculation")
    st.success(
        "This is **not a fixed example**. Every numerical value below is generated from the currently active "
        "simulation case. Change the sidebar inputs, press **Run simulation**, and this complete walkthrough "
        "updates together with the main results."
    )
    st.caption(
        "The equations stay the same; the substituted values, dimensionless groups, transfer coefficients, "
        "ODE solution, pressure drop and flooding calculation change with the selected operating conditions."
    )

    selected_packing = PACKING_DATA[active.packing]
    rhoL = result["water_properties"]["rho"]
    muL = result["water_properties"]["mu"]
    sigmaL = result["water_properties"]["sigma"]
    a_t_live = selected_packing["a"]
    dp_live = selected_packing["d"]
    eps_live = selected_packing["epsilon"]
    sigma_c_live = selected_packing["sigma_c"]
    psi_live = selected_packing["psi"]

    # Onda groups not all exposed directly by the engine are reproduced here
    # from exactly the same active inputs and equations used by the model.
    ReL_live = L_mass_flux / (a_t_live * muL)
    ReG_live = G_mass_flux / (a_t_live * method_sim.mu_G)
    FrL_live = L_mass_flux**2 * a_t_live / (rhoL**2 * 9.81)
    WeL_live = L_mass_flux**2 / (rhoL * sigmaL * a_t_live)
    wet_exp_live = (
        -1.45
        * (sigma_c_live / sigmaL)**0.75
        * max(ReL_live, 1e-30)**0.10
        * max(FrL_live, 1e-30)**(-0.05)
        * max(WeL_live, 1e-30)**0.20
    )

    current_input_df = pd.DataFrame([
        ["Packing", active.packing, "-"],
        ["Column diameter, D", active.diameter_m, "m"],
        ["Packed height, Z", active.packed_height_m, "m"],
        ["Water flow", active.liquid_kg_h, "kg/h"],
        ["Entered gas flow", active.gas_m3_h, "m³/h"],
        ["Gas-flow basis", active.flow_reference, "-"],
        ["Temperature", active.temperature_C, "°C"],
        ["Pressure", active.pressure_Pa / 1e5, "bar abs"],
        ["Inlet concentration", active.inlet_concentration, active.concentration_basis],
        ["ACN mass fraction", 100*active.acn_mass_fraction, "%"],
        ["VAc mass fraction", 100*active.vac_mass_fraction, "%"],
        ["Target outlet", active.target_concentration, active.concentration_basis],
    ], columns=["Active input", "Value", "Unit / basis"])
    st.dataframe(current_input_df, use_container_width=True, hide_index=True)

    with st.expander("Live Step 1 — Column geometry and flow conversion", expanded=True):
        st.markdown("#### 1A. Column area")
        st.latex(r"A_c=\frac{\pi D^2}{4}")
        st.latex(
            rf"A_c=\frac{{\pi({active.diameter_m:.6g})^2}}{{4}}="
            rf"{area_col:.8g}\;\mathrm{{m^2}}"
        )

        st.markdown("#### 1B. Gas-flow basis")
        if active.flow_reference.lower().startswith("normal"):
            st.latex(r"Q_{act}=Q_N\left(\frac{T}{T_N}\right)\left(\frac{P_N}{P}\right)")
            st.latex(
                rf"Q_{{act}}={active.gas_m3_h:.6g}"
                rf"\left(\frac{{{method_sim.T:.6g}}}{{273.15}}\right)"
                rf"\left(\frac{{101325}}{{{active.pressure_Pa:.7g}}}\right)"
                rf"={q_actual:.8g}\;\mathrm{{m^3/h}}"
            )
        else:
            st.write(
                f"Gas flow is entered as **Actual**, therefore no normal-to-actual conversion is applied: "
                f"Q_act = **{q_actual:.8g} m³/h**."
            )

        st.markdown("#### 1C. Molar flows, mass fluxes and superficial velocity")
        st.latex(r"\dot n_G=\frac{Q_{act}P}{RT}\frac{1}{3600}")
        st.latex(
            rf"\dot n_G=\frac{{({q_actual:.6g})({active.pressure_Pa:.7g})}}"
            rf"{{(8.314462618)({method_sim.T:.6g})}}\frac{{1}}{{3600}}"
            rf"={nG:.8g}\;\mathrm{{mol/s}}"
        )
        st.latex(r"\dot n_L=\frac{\dot m_L}{MW_{H_2O}}\frac{1}{3600}")
        st.latex(
            rf"\dot n_L=\frac{{{active.liquid_kg_h:.7g}}}{{0.018015}}\frac{{1}}{{3600}}"
            rf"={nL:.8g}\;\mathrm{{mol/s}}"
        )
        st.latex(r"G'=\frac{Q_{act}\rho_G}{3600A_c},\qquad L'=\frac{\dot m_L}{3600A_c},\qquad U_G=\frac{Q_{act}}{3600A_c}")
        st.latex(
            rf"G'={G_mass_flux:.8g}\;\mathrm{{kg/(m^2s)}},\quad "
            rf"L'={L_mass_flux:.8g}\;\mathrm{{kg/(m^2s)}},\quad "
            rf"U_G={Ug:.8g}\;\mathrm{{m/s}}"
        )

    with st.expander("Live Step 2 — Concentration-basis conversion to component mole fractions", expanded=True):
        CgasN = method_sim.normal_gas_molar_concentration()
        st.latex(r"C_{G,N}=\frac{P_N}{RT_N}")
        st.latex(
            rf"C_{{G,N}}=\frac{{101325}}{{(8.314462618)(273.15)}}"
            rf"={CgasN:.8g}\;\mathrm{{mol/Nm^3}}"
        )
        st.write(
            f"The selected inlet basis is **{active.concentration_basis}**. The model converts that total mixture "
            "concentration to ACN and VAc gas-phase mole fractions before solving mass transfer."
        )
        inlet_y_df = pd.DataFrame([
            ["ACN", result["ACN"]["y_in"], 1e6*result["ACN"]["y_in"], result["ACN_in"]],
            ["VAc", result["VAc"]["y_in"], 1e6*result["VAc"]["y_in"], result["VAc_in"]],
        ], columns=["Component", "y_in (mol/mol)", "Equivalent component ppmv", f"Component inlet ({unit})"])
        st.dataframe(inlet_y_df, use_container_width=True, hide_index=True)
        st.write("Equivalent total inlet values in every supported basis:")
        st.dataframe(pd.DataFrame([
            ["mgVOC/Nm³", result["inlet_by_basis"]["mgVOC/Nm³"]],
            ["mgC/Nm³", result["inlet_by_basis"]["mgC/Nm³"]],
            ["ppmv", result["inlet_by_basis"]["ppmv"]],
        ], columns=["Basis", "Current physical inlet"]), use_container_width=True, hide_index=True)

    worked_component = st.selectbox(
        "Component for detailed live mass-transfer substitution",
        ["ACN", "VAc"],
        key="live_worked_component",
        help="Both components are solved every run. This selector only chooses which one is expanded line-by-line below.",
    )
    wc = result[worked_component]["coefficients"]
    voc_data = {
        "ACN": {"MW": 53.06e-3, "DL": 1.0e-9, "DG": 1.0e-5},
        "VAc": {"MW": 86.09e-3, "DL": 0.9e-9, "DG": 0.9e-5},
    }[worked_component]

    with st.expander(f"Live Step 3 — {worked_component}: properties, Henry equilibrium and Onda dimensionless groups", expanded=True):
        st.markdown("#### 3A. Bulk gas and water properties")
        st.latex(r"\rho_G=\frac{PMW_G}{RT}")
        st.latex(
            rf"\rho_G=\frac{{({active.pressure_Pa:.7g})(0.029)}}"
            rf"{{(8.314462618)({method_sim.T:.6g})}}={method_sim.rho_G:.8g}\;\mathrm{{kg/m^3}}"
        )
        st.write(
            f"At **{active.temperature_C:.4g} °C**, water properties used are "
            f"ρL = **{rhoL:.7g} kg/m³**, μL = **{muL:.7g} Pa·s**, "
            f"σL = **{sigmaL:.7g} N/m**."
        )

        st.markdown(f"#### 3B. {worked_component} Henry equilibrium")
        st.latex(r"y^*=mx")
        st.latex(r"m=\frac{H(\rho_L/MW_{H_2O})}{P}")
        st.latex(
            rf"m_{{{worked_component}}}=\frac{{({wc['H']:.7g})"
            rf"({rhoL:.7g}/0.018015)}}{{{active.pressure_Pa:.7g}}}"
            rf"={wc['m_x']:.8g}"
        )
        st.write(f"Current Henry constant: **H = {wc['H']:.7g} Pa·m³/mol**.")

        st.markdown("#### 3C. Onda dimensionless groups")
        st.latex(r"Re_L=\frac{L'}{a_t\mu_L},\quad Re_G=\frac{G'}{a_t\mu_G}")
        st.latex(
            rf"Re_L=\frac{{{L_mass_flux:.7g}}}{{({a_t_live:.7g})({muL:.7g})}}={ReL_live:.8g}"
        )
        st.latex(
            rf"Re_G=\frac{{{G_mass_flux:.7g}}}{{({a_t_live:.7g})({method_sim.mu_G:.7g})}}={ReG_live:.8g}"
        )
        st.latex(r"Fr_L=\frac{L'^2a_t}{\rho_L^2g},\qquad We_L=\frac{L'^2}{\rho_L\sigma_La_t}")
        st.latex(rf"Fr_L={FrL_live:.8g},\qquad We_L={WeL_live:.8g}")
        st.latex(r"Sc_L=\frac{\mu_L}{\rho_LD_L},\qquad Sc_G=\frac{\mu_G}{\rho_GD_G}")
        st.latex(rf"Sc_L={wc['Sc_L']:.8g},\qquad Sc_G={wc['Sc_G']:.8g}")

    with st.expander(f"Live Step 4 — {worked_component}: wetted area, kL, kG and overall KG", expanded=True):
        st.markdown("#### 4A. Effective wetted area")
        st.latex(
            r"\frac{a_e}{a_t}=1-\exp\left[-1.45"
            r"\left(\frac{\sigma_c}{\sigma_L}\right)^{0.75}"
            r"Re_L^{0.10}Fr_L^{-0.05}We_L^{0.20}\right]"
        )
        st.latex(
            rf"E_{{wet}}={wet_exp_live:.8g},\qquad "
            rf"a_e={a_t_live:.7g}[1-\exp({wet_exp_live:.8g})]"
            rf"={wc['a_e']:.8g}\;\mathrm{{m^2/m^3}}"
        )
        st.write(
            f"Wetting fraction = **{100*wc['wetting_fraction']:.3f}%** of nominal packing area "
            f"(a_t = {a_t_live:.7g} m²/m³)."
        )

        st.markdown("#### 4B. Liquid-film coefficient")
        st.latex(
            r"k_L=0.0051\left(\frac{\mu_Lg}{\rho_L}\right)^{1/3}"
            r"\left(\frac{L'}{a_e\mu_L}\right)^{2/3}"
            r"Sc_L^{-1/2}(a_td_p)^{0.4}"
        )
        st.latex(rf"k_L={wc['k_L']:.8g}\;\mathrm{{m/s}}")

        st.markdown("#### 4C. Gas-film coefficient")
        st.latex(
            r"k_G=5.23\left(\frac{a_tD_G}{RT}\right)"
            r"Re_G^{0.7}Sc_G^{1/3}(a_td_p)^{-2}"
        )
        st.latex(rf"k_G={wc['k_G']:.8g}\;\mathrm{{mol/(m^2\,s\,Pa)}}")

        st.markdown("#### 4D. Overall gas-side coefficient")
        st.latex(r"\frac{1}{K_G}=\frac{1}{k_G}+\frac{H}{k_L}")
        st.latex(
            rf"\frac{{1}}{{K_G}}=\frac{{1}}{{{wc['k_G']:.7g}}}+"
            rf"\frac{{{wc['H']:.7g}}}{{{wc['k_L']:.7g}}}"
        )
        st.latex(rf"K_G={wc['K_G']:.8g}\;\mathrm{{mol/(m^2\,s\,Pa)}}")
        st.write(
            f"Resistance split: **gas film {100*wc['gas_resistance_fraction']:.2f}%**, "
            f"**liquid film {100*wc['liquid_resistance_fraction']:.2f}%**."
        )

    with st.expander(f"Live Step 5 — {worked_component}: absorption factor, HTU and NTU", expanded=True):
        st.markdown("#### 5A. Absorption factor")
        st.latex(r"A=\frac{L}{mG}")
        st.latex(
            rf"A_{{{worked_component}}}=\frac{{{nL:.8g}}}"
            rf"{{({wc['m_x']:.8g})({nG:.8g})}}={wc['absorption_factor']:.8g}"
        )

        Gflux_mol_live = nG / area_col
        st.markdown("#### 5B. Overall gas-phase HTU")
        st.latex(r"HTU_{OG}=\frac{G_{mol}'}{K_Ga_eP}")
        st.latex(
            rf"HTU_{{OG}}=\frac{{{Gflux_mol_live:.8g}}}"
            rf"{{({wc['K_G']:.8g})({wc['a_e']:.8g})({active.pressure_Pa:.7g})}}"
            rf"={wc['HTU_OG_m']:.8g}\;\mathrm{{m}}"
        )
        st.markdown("#### 5C. Diagnostic NTU")
        st.latex(r"NTU_{OG}=\frac{Z}{HTU_{OG}}")
        st.latex(
            rf"NTU_{{OG}}=\frac{{{active.packed_height_m:.7g}}}{{{wc['HTU_OG_m']:.8g}}}"
            rf"={wc['NTU_OG']:.8g}"
        )
        st.caption(
            "This NTU is a diagnostic Z/HTU value. The actual outlet is still calculated by the counter-current ODE solution, not by an NTU shortcut."
        )

    with st.expander(f"Live Step 6 — {worked_component}: counter-current ODE solution and outlet", expanded=True):
        Gflux_mol = nG / area_col
        Lflux_mol = nL / area_col
        rate_const = wc["K_G"] * wc["a_e"] * active.pressure_Pa
        st.latex(r"r=K_Ga_eP(y-mx)")
        st.latex(
            rf"K_Ga_eP=({wc['K_G']:.8g})({wc['a_e']:.8g})({active.pressure_Pa:.7g})"
            rf"={rate_const:.8g}\;\mathrm{{s^{{-1}}\,mol/m^3\;basis}}"
        )
        st.latex(r"\frac{dy}{dz}=-\frac{r}{G_{mol}'},\qquad \frac{dx}{dz}=-\frac{r}{L_{mol}'}")
        st.write(
            f"Molar fluxes used by the ODE: G′ = **{Gflux_mol:.8g} mol/m²·s**, "
            f"L′ = **{Lflux_mol:.8g} mol/m²·s**."
        )
        st.write(
            f"Boundary conditions for {worked_component}: y(0) = **{result[worked_component]['y_in']:.8g}**, "
            f"and fresh-water condition x(Z) = **0**. The shooting method solved "
            f"x(0) = **{result[worked_component]['x_bottom']:.8g}**."
        )
        component_out_selected = method_sim.y_to_component_concentration(
            result[worked_component]["y_out"], worked_component, active.concentration_basis
        )
        st.latex(
            rf"y_{{out,{worked_component}}}={result[worked_component]['y_out']:.8g}"
        )
        st.write(
            f"This corresponds to **{component_out_selected:.8g} {active.concentration_basis}** for "
            f"{worked_component}, with **{100*result[worked_component]['removal']:.4f}% removal**."
        )

    with st.expander("Live Step 7 — Recombine ACN + VAc and solve required packed height", expanded=True):
        live_comp_df = pd.DataFrame([
            ["ACN", result["ACN_in"], result["ACN_out"], 100*result["ACN_removal"], result["ACN"]["coefficients"]["absorption_factor"], result["ACN"]["coefficients"]["HTU_OG_m"], result["ACN"]["coefficients"]["NTU_OG"]],
            ["VAc", result["VAc_in"], result["VAc_out"], 100*result["VAc_removal"], result["VAc"]["coefficients"]["absorption_factor"], result["VAc"]["coefficients"]["HTU_OG_m"], result["VAc"]["coefficients"]["NTU_OG"]],
        ], columns=["Component", f"In ({unit})", f"Out ({unit})", "Removal %", "A", "HTUOG (m)", "NTUOG"])
        st.dataframe(live_comp_df, use_container_width=True, hide_index=True)
        st.latex(r"C_{out,total}=C_{out,ACN}+C_{out,VAc}")
        st.latex(
            rf"C_{{out,total}}={result['ACN_out']:.8g}+{result['VAc_out']:.8g}"
            rf"={result['outlet_concentration']:.8g}\;\mathrm{{{unit.replace('³','^3')}}}"
        )
        st.latex(r"\eta=1-\frac{C_{out,total}}{C_{in,total}}")
        st.write(f"Overall removal = **{100*result['overall_removal']:.5f}%**.")
        st.write(
            "For required height, the model repeatedly re-solves both component ODEs while varying Z and uses "
            "Brent's root method on C_out(Z) − C_target = 0."
        )
        if z_req is None:
            st.warning("For the current target, the required packed height was not reached within the 20 m search limit.")
        else:
            st.latex(
                rf"C_{{out}}(Z)-{active.target_concentration:.8g}=0\quad\Rightarrow\quad "
                rf"Z_{{required}}={z_req:.8g}\;\mathrm{{m}}"
            )

    with st.expander("Live Step 8 — Pressure drop and GPDC flooding capacity", expanded=True):
        hL_live = result["hL"]
        dPdry_live = result["dPdry_Pa_m"]
        dPwet_live = result["dPwet_Pa_m"]
        st.markdown("#### 8A. Liquid holdup and pressure drop")
        st.latex(r"h_L=\left[\frac{12\mu_L(L'/\rho_L)a_t^2}{g\rho_L}\right]^{1/3}")
        st.latex(rf"h_L={hL_live:.8g}")
        st.latex(r"\left(\frac{\Delta P}{Z}\right)_{dry}=\psi\frac{a_t}{\varepsilon^3}\frac{\rho_GU_G^2}{2}")
        st.latex(
            rf"(\Delta P/Z)_{{dry}}={psi_live:.6g}\frac{{{a_t_live:.6g}}}{{{eps_live:.6g}^3}}"
            rf"\frac{{({method_sim.rho_G:.6g})({Ug:.6g})^2}}{{2}}={dPdry_live:.8g}\;\mathrm{{Pa/m}}"
        )
        st.latex(r"\left(\frac{\Delta P}{Z}\right)_{wet}=\left(\frac{\Delta P}{Z}\right)_{dry}\left(\frac{\varepsilon}{\varepsilon-h_L}\right)^3")
        st.write(
            f"Wet packed-bed pressure drop = **{result['dPwet_mbar_m']:.8g} mbar/m**; "
            f"total packed-bed ΔP = **{result['Total_dP_mbar']:.8g} mbar**."
        )

        st.markdown("#### 8B. GPDC flooding solution")
        st.latex(r"F_{LV}=\frac{L'}{G'_{flood}}\sqrt{\frac{\rho_G}{\rho_L}}")
        st.latex(r"C_S=U_{G,flood}\sqrt{\frac{\rho_G}{\rho_L-\rho_G}},\qquad C_P=C_S\sqrt{F_P}\nu_L^{0.05}")
        st.write(
            f"Packing factor used: **Fp = {result['Fp_ft']:.8g} ft⁻¹** ({result['Fp_basis']}). "
            f"The solved GPDC values are F_LV = **{result['F_LV_flood']:.8g}**, "
            f"CP = **{result['CP_flood']:.8g}**, and U_G,flood = **{result['U_flood']:.8g} m/s**."
        )
        st.latex(r"\%Flood=100\frac{U_{G,oper}}{U_{G,flood}}")
        st.latex(
            rf"\%Flood=100\frac{{{result['Ug']:.8g}}}{{{result['U_flood']:.8g}}}"
            rf"={result['flooding_percent']:.8g}\%"
        )
        st.write(f"Hydraulic regime: **{result['hydraulic_regime']}**")

    with st.expander("Live Step 9 — Final calculation summary", expanded=True):
        live_summary = pd.DataFrame([
            ["Outlet concentration", result["outlet_concentration"], unit],
            ["Overall removal", 100*result["overall_removal"], "%"],
            ["Required packed height", "Not reached" if z_req is None else z_req, "m"],
            ["Wet pressure drop", result["dPwet_mbar_m"], "mbar/m"],
            ["Total packed-bed pressure drop", result["Total_dP_mbar"], "mbar"],
            ["Operating gas velocity", result["Ug"], "m/s"],
            ["Flood gas velocity", result["U_flood"], "m/s"],
            ["Fraction of flood", result["flooding_percent"], "%"],
        ], columns=["Final quantity", "Current result", "Unit"])
        st.dataframe(live_summary, use_container_width=True, hide_index=True)
        if result["warnings"]:
            st.write("Diagnostics triggered by this exact case:")
            for w in result["warnings"]:
                st.warning(w)
        else:
            st.success("This current case does not trigger any programmed diagnostic warning.")

    st.divider()
    st.subheader("Theory & correlation reference")
    st.caption(
        "The sections below explain the equations and assumptions independent of any single operating case. "
        "The Live Worked Example above uses the same equations with the active numerical inputs substituted automatically."
    )

    with st.expander("1. Geometry, flow basis and the first calculations", expanded=True):
        st.markdown(
            """
### 1.1 Column cross-sectional area
For a cylindrical packed column:
            """
        )
        st.latex(r"A_c=\frac{\pi D^2}{4}")
        st.write(
            f"For the current case, **D = {active.diameter_m:.4g} m**, therefore "
            f"**A_c = {area_col:.6f} m²**."
        )

        st.markdown("### 1.2 Actual vs normal gas flow")
        st.write(
            "If the entered gas flow is marked **Normal**, the model converts it from the model's normal state "
            "(0 °C, 1 atm) to the selected operating temperature and pressure using the ideal-gas relation. "
            "If **Aktual** is selected, the entered value is used directly as operating volumetric flow."
        )
        st.latex(r"Q_{act}=Q_N\left(\frac{T}{T_N}\right)\left(\frac{P_N}{P}\right)")
        st.write(
            f"Current input basis: **{active.flow_reference}**  |  "
            f"Entered Q = **{active.gas_m3_h:.4g} m³/h**  |  "
            f"Operating Q used by the model = **{q_actual:.4g} m³/h**"
        )

        st.markdown("### 1.3 Molar flow, mass flux and superficial gas velocity")
        st.latex(r"\dot n_G=\frac{Q_{act}P}{RT}\frac{1}{3600}")
        st.latex(r"\dot n_L=\frac{\dot m_L}{MW_{H_2O}}\frac{1}{3600}")
        st.latex(r"G'=\frac{\dot m_G}{A_c}=\frac{Q_{act}\rho_G}{3600A_c}")
        st.latex(r"L'=\frac{\dot m_L}{3600A_c}")
        st.latex(r"U_G=\frac{Q_{act}}{3600A_c}")
        flow_df = pd.DataFrame({
            "Quantity": ["Gas molar flow", "Liquid molar flow", "Gas mass flux", "Liquid mass flux", "Superficial gas velocity"],
            "Current value": [nG, nL, G_mass_flux, L_mass_flux, Ug],
            "Unit": ["mol/s", "mol/s", "kg/m²·s", "kg/m²·s", "m/s"],
        })
        st.dataframe(flow_df, use_container_width=True, hide_index=True)

    with st.expander("2. Concentration basis: mgVOC/Nm³, mgC/Nm³ and ppmv"):
        st.markdown(
            """
The internal transport calculation is performed with **gas-phase mole fraction, y**. The selected concentration basis is therefore converted to component mole fractions before the column calculation begins.

The model defines normal conditions as:
- **T_N = 273.15 K (0 °C)**
- **P_N = 101325 Pa (1 atm)**

The normal-gas molar concentration is:
            """
        )
        st.latex(r"C_{G,N}=\frac{P_N}{RT_N}\quad [mol/Nm^3]")
        st.write(f"Current model value: **{method_sim.normal_gas_molar_concentration():.5f} mol/Nm³**")

        st.markdown("#### A) Input entered as mgVOC/Nm³")
        st.latex(r"C_{i,VOC}=C_{VOC,total}\,w_i")
        st.latex(r"y_i=\frac{C_{i,VOC}\,10^{-6}/MW_i}{C_{G,N}}")
        st.caption("Here w_i is the entered VOC **mass fraction**, not mole fraction.")

        st.markdown("#### B) Input entered as mgC/Nm³")
        st.latex(r"w_{C,i}=\frac{N_{C,i}MW_C}{MW_i}")
        st.latex(r"w_{C,mix}=\sum_i w_iw_{C,i}")
        st.latex(r"C_{VOC,total}=\frac{C_{C,total}}{w_{C,mix}}")
        st.write("The resulting total VOC mass concentration is then split by the entered ACN/VAc mass fractions and converted to y_i.")

        st.markdown("#### C) Input entered as ppmv")
        st.write("Because ACN/VAc composition is still entered on a **mass basis**, the model first converts mass fractions to VOC molar shares:")
        st.latex(r"z_i=\frac{w_i/MW_i}{\sum_j w_j/MW_j}")
        st.latex(r"y_i=C_{ppmv,total}\times10^{-6}\,z_i")

        st.markdown("#### Output conversion")
        st.latex(r"C_{i,VOC}[mg/Nm^3]=y_iC_{G,N}MW_i\times10^6")
        st.latex(r"C_{i,C}[mgC/Nm^3]=y_iC_{G,N}N_{C,i}MW_C\times10^6")
        st.latex(r"C_{i,ppmv}=y_i\times10^6")

        basis_df = pd.DataFrame([
            ["mgVOC/Nm³", result["inlet_by_basis"]["mgVOC/Nm³"], result["outlet_by_basis"]["mgVOC/Nm³"]],
            ["mgC/Nm³", result["inlet_by_basis"]["mgC/Nm³"], result["outlet_by_basis"]["mgC/Nm³"]],
            ["ppmv", result["inlet_by_basis"]["ppmv"], result["outlet_by_basis"]["ppmv"]],
        ], columns=["Basis", "Current inlet", "Current outlet"])
        st.dataframe(basis_df, use_container_width=True, hide_index=True)

    with st.expander("3. Gas properties, water properties and Henry equilibrium"):
        st.markdown("### 3.1 Gas density — ideal-gas approximation")
        st.latex(r"\rho_G=\frac{PMW_G}{RT}")
        st.write(
            "The carrier gas is represented by an air-like molecular weight of **0.029 kg/mol**. "
            "VOC loading is assumed dilute enough that it does not materially change bulk gas density."
        )

        st.markdown("### 3.2 Gas viscosity — Sutherland equation")
        st.latex(r"\mu_G=\mu_{ref}\left(\frac{T}{T_{ref}}\right)^{3/2}\frac{T_{ref}+S}{T+S}")
        st.write("Implemented constants: μ_ref = 1.716×10⁻⁵ Pa·s, T_ref = 273.15 K, S = 110.4 K.")

        st.markdown("### 3.3 Pure-water density")
        st.latex(r"\rho_L=1000\left[1-\frac{T_C+288.9414}{508929.2(T_C+68.12963)}(T_C-3.9863)^2\right]")

        st.markdown("### 3.4 Pure-water dynamic viscosity")
        st.latex(r"\mu_L=2.414\times10^{-5}\,10^{247.8/(T_K-140)}")

        st.markdown("### 3.5 Pure-water surface tension")
        st.latex(r"\tau=1-\frac{T}{T_c}")
        st.latex(r"\sigma_L=0.2358\,\tau^{1.256}(1-0.625\tau)")
        st.write("The surface-tension expression is the IAPWS form with T_c = 647.096 K.")

        st.markdown("### 3.6 Henry constant and its temperature correction")
        st.latex(r"H_i(T)=H_{i,25}\exp\left[\left(\frac{\Delta H_H}{R}\right)_i\left(\frac{1}{298.15}-\frac{1}{T}\right)\right]")
        st.write("The stored H25 values are in atm·m³/mol and are converted internally to Pa·m³/mol.")
        st.latex(r"y_i^*=m_ix_i")
        st.latex(r"m_i=\frac{H_i\,(\rho_L/MW_{H_2O})}{P}")
        st.write(
            "Thus the model assumes a **linear Henry-law equilibrium** and an effectively dilute liquid phase. "
            "Activity coefficients and concentration-dependent Henry constants are not included."
        )

        props_df = pd.DataFrame([
            ["Gas density", method_sim.rho_G, "kg/m³"],
            ["Gas viscosity", method_sim.mu_G, "Pa·s"],
            ["Water density", result["water_properties"]["rho"], "kg/m³"],
            ["Water viscosity", result["water_properties"]["mu"], "Pa·s"],
            ["Water surface tension", result["water_properties"]["sigma"], "N/m"],
            ["ACN Henry constant", result["ACN"]["coefficients"]["H"], "Pa·m³/mol"],
            ["VAc Henry constant", result["VAc"]["coefficients"]["H"], "Pa·m³/mol"],
        ], columns=["Property", "Current value", "Unit"])
        st.dataframe(props_df, use_container_width=True, hide_index=True)

    with st.expander("4. Onda mass-transfer model: dimensionless groups, wetted area, kL and kG"):
        st.markdown(
            """
The mass-transfer section follows the **Onda–Takeuchi–Okumoto style correlations for random packed columns**. The model uses the packing's nominal specific area `a_t`, nominal size `d_p`, and critical surface tension `σ_c` together with gas/liquid loading and fluid properties.

### 4.1 Dimensionless groups used by the implemented correlation
            """
        )
        st.latex(r"Re_L=\frac{L'}{a_t\mu_L}")
        st.latex(r"Re_G=\frac{G'}{a_t\mu_G}")
        st.latex(r"Fr_L=\frac{L'^2a_t}{\rho_L^2g}")
        st.latex(r"We_L=\frac{L'^2}{\rho_L\sigma_La_t}")
        st.latex(r"Sc_L=\frac{\mu_L}{\rho_LD_L}")
        st.latex(r"Sc_G=\frac{\mu_G}{\rho_GD_G}")

        st.markdown("### 4.2 Effective wetted interfacial area")
        st.latex(
            r"\frac{a_e}{a_t}=1-\exp\left[-1.45\left(\frac{\sigma_c}{\sigma_L}\right)^{0.75}"
            r"Re_L^{0.10}Fr_L^{-0.05}We_L^{0.20}\right]"
        )
        st.write(
            "The code limits a_e to the physical interval 0 < a_e ≤ a_t. The resulting a_e is treated as the "
            "gas–liquid interfacial area available for mass transfer."
        )

        st.markdown("### 4.3 Liquid-film coefficient")
        st.latex(
            r"k_L=0.0051\left(\frac{\mu_Lg}{\rho_L}\right)^{1/3}"
            r"\left(\frac{L'}{a_e\mu_L}\right)^{2/3}Sc_L^{-1/2}(a_td_p)^{0.4}"
        )

        st.markdown("### 4.4 Gas-film coefficient")
        st.latex(
            r"k_G=5.23\left(\frac{a_tD_G}{RT}\right)Re_G^{0.7}Sc_G^{1/3}(a_td_p)^{-2}"
        )
        st.write(
            "In this implementation k_G is on a gas-side pressure-driving-force basis, allowing it to be "
            "combined with H/k_L in the overall resistance equation below."
        )

        st.markdown("### 4.5 Overall gas-side coefficient — two-film resistance addition")
        st.latex(r"\frac{1}{K_G}=\frac{1}{k_G}+\frac{H}{k_L}")
        st.latex(r"K_G=\left(\frac{1}{k_G}+\frac{H}{k_L}\right)^{-1}")
        st.write("The model also reports the fraction of total resistance assigned to each film:")
        st.latex(r"f_G=\frac{1/k_G}{1/K_G},\qquad f_L=\frac{H/k_L}{1/K_G}")

        onda_rows = []
        for comp in ("ACN", "VAc"):
            c = result[comp]["coefficients"]
            onda_rows.append({
                "Component": comp,
                "Re_L": c["Re_L"], "Re_G": c["Re_G"], "Sc_L": c["Sc_L"], "Sc_G": c["Sc_G"],
                "a_t": c["a_t"], "a_e": c["a_e"], "Wetting %": 100*c["wetting_fraction"],
                "k_L": c["k_L"], "k_G": c["k_G"], "K_G": c["K_G"], "m": c["m_x"],
            })
        st.dataframe(pd.DataFrame(onda_rows), use_container_width=True, hide_index=True)

    with st.expander("5. Absorption factor, HTU and NTU diagnostics"):
        st.markdown("### 5.1 Absorption factor")
        st.latex(r"A_i=\frac{L}{m_iG}")
        st.write(
            "L and G here are total **molar flow rates** of liquid water and gas. A is component-specific because "
            "m differs strongly between ACN and VAc. As a qualitative diagnostic, larger A favors absorption; "
            "A ≤ 1 triggers a warning in the current code."
        )

        st.markdown("### 5.2 Overall-gas-phase HTU")
        st.latex(r"HTU_{OG,i}=\frac{G'}{K_{G,i}a_{e,i}P}")
        st.markdown("### 5.3 NTU diagnostic")
        st.latex(r"NTU_{OG,i}=\frac{Z}{HTU_{OG,i}}")
        st.warning(
            "The displayed NTU is a **diagnostic Z/HTU value**. It is not independently integrated from a "
            "variable operating-line integral. The actual outlet concentration is obtained from the ODE model, not "
            "from an exponential NTU shortcut."
        )
        antu_df = pd.DataFrame([
            [comp, result[comp]["coefficients"]["absorption_factor"], result[comp]["coefficients"]["HTU_OG_m"], result[comp]["coefficients"]["NTU_OG"]]
            for comp in ("ACN", "VAc")
        ], columns=["Component", "Absorption factor A", "HTU_OG (m)", "NTU_OG"])
        st.dataframe(antu_df, use_container_width=True, hide_index=True)

    with st.expander("6. Counter-current differential column model — how outlet concentration is actually solved"):
        st.markdown(
            """
The packed section is treated as a **steady-state counter-current differential absorber**. Coordinate `z = 0` is the gas inlet / liquid outlet end (bottom), and `z = Z` is the gas outlet / fresh-liquid inlet end (top).

For each component, the local overall molar transfer rate per packed volume is:
            """
        )
        st.latex(r"r_i=K_{G,i}a_{e,i}P\,(y_i-m_ix_i)")
        st.write("where the local driving force is y_i − y_i* = y_i − m_i x_i.")

        st.markdown("The model then integrates:")
        st.latex(r"\frac{dy_i}{dz}=-\frac{r_i}{G'}")
        st.latex(r"\frac{dx_i}{dz}=-\frac{r_i}{L'}")
        st.write(
            "G′ and L′ are gas and liquid molar fluxes [mol/m²·s]. The negative signs are consistent with the "
            "chosen upward z coordinate: both y and x decrease from the bottom toward the top."
        )

        st.markdown("### Boundary conditions and numerical solution")
        st.latex(r"y_i(0)=y_{i,in}")
        st.latex(r"x_i(Z)=0")
        st.write(
            "Fresh inlet water is therefore assumed to contain **zero ACN and zero VAc**. The unknown bottom-liquid "
            "composition x_i(0) is found by a shooting method: the model guesses x_i(0), integrates the ODEs, "
            "and uses Brent's root solver until x_i(Z)=0 is satisfied."
        )
        st.code(
            "1) Guess x_bottom\n"
            "2) Integrate [y, x] from z=0 to z=Z with solve_ivp\n"
            "3) Evaluate error = x(Z)\n"
            "4) Use brentq to drive error to zero\n"
            "5) Return y(Z) as gas outlet mole fraction",
            language="text",
        )
        st.write(
            "SciPy `solve_ivp` is used with rtol = 1×10⁻⁸, atol = 1×10⁻¹² and a maximum step of about Z/100. "
            "If the counter-current boundary condition cannot be bracketed, the model raises an error rather than silently returning a fallback result."
        )

        ode_df = pd.DataFrame([
            ["ACN", result["ACN"]["y_in"], result["ACN"]["y_out"], result["ACN"]["x_bottom"], 100*result["ACN"]["removal"]],
            ["VAc", result["VAc"]["y_in"], result["VAc"]["y_out"], result["VAc"]["x_bottom"], 100*result["VAc"]["removal"]],
        ], columns=["Component", "y_in", "y_out", "x_bottom solved", "Removal %"])
        st.dataframe(ode_df, use_container_width=True, hide_index=True)

    with st.expander("7. Total outlet concentration, removal and required packed height"):
        st.markdown("After ACN and VAc are solved independently, their outlet mole fractions are converted back to the selected concentration basis and summed.")
        st.latex(r"C_{out,total}=C_{out,ACN}+C_{out,VAc}")
        st.latex(r"\eta=1-\frac{C_{out,total}}{C_{in,total}}")
        st.markdown("### Required height")
        st.write(
            "For a specified target concentration, the model repeatedly solves the complete component ODE problem at different packed heights and finds the root of:"
        )
        st.latex(r"f(Z)=C_{out}(Z)-C_{target}=0")
        st.write(
            "Brent's method is used between approximately zero height and **Zmax = 20 m**. If the target is still not reached at 20 m, the model returns `None` / 'Not reached'."
        )
        st.write(
            f"Current result: outlet = **{result['outlet_concentration']:.5g} {unit}**, removal = "
            f"**{100*result['overall_removal']:.3f}%**, required height = "
            + ("**Not reached within 20 m**" if z_req is None else f"**{z_req:.4g} m**")
        )

    with st.expander("8. Pressure-drop screening model"):
        st.warning(
            "This section is intentionally labelled **screening-level**. It is not a complete vendor hydraulic-rating "
            "model and is not presented as a full Billet–Schultes implementation."
        )
        st.markdown("### 8.1 Liquid holdup approximation")
        st.latex(r"h_L=\left[\frac{12\mu_L(L'/\rho_L)a_t^2}{g\rho_L}\right]^{1/3}")
        st.write("The code limits h_L to a maximum of 0.90 ε.")

        st.markdown("### 8.2 Dry-bed pressure drop")
        st.latex(r"\left(\frac{\Delta P}{Z}\right)_{dry}=\psi\frac{a_t}{\varepsilon^3}\frac{\rho_GU_G^2}{2}")
        st.write("ψ is a packing-specific empirical geometry multiplier stored in the packing database.")

        st.markdown("### 8.3 Wet-bed correction")
        st.latex(r"\left(\frac{\Delta P}{Z}\right)_{wet}=\left(\frac{\Delta P}{Z}\right)_{dry}\left(\frac{\varepsilon}{\varepsilon-h_L}\right)^3")
        st.latex(r"\Delta P_{total}=\left(\frac{\Delta P}{Z}\right)_{wet}Z")
        st.write(
            f"Current prediction: **{result['dPwet_mbar_m']:.5f} mbar/m**, total packed-bed ΔP = "
            f"**{result['Total_dP_mbar']:.5f} mbar**."
        )

    with st.expander("9. Flooding capacity — GPDC implementation"):
        st.markdown(
            """
Flooding is **not** calculated from operating pressure drop. The code independently solves a GPDC-style flooding velocity and then compares the actual superficial gas velocity with that flood velocity.

### 9.1 Packing factor
For packings with a stored empirical packing factor, that value is used directly. If no empirical factor exists, the code uses a clearly flagged geometric fallback:
            """
        )
        st.latex(r"F_P\approx\frac{a_t/\varepsilon^3}{3.28084}\quad [ft^{-1}]")
        st.write("The division by 3.28084 converts the geometric 1/m value to 1/ft.")

        st.markdown("### 9.2 GPDC flow parameter at flood")
        st.latex(r"F_{LV}=\frac{L'}{G'_{flood}}\sqrt{\frac{\rho_G}{\rho_L}}")
        st.latex(r"G'_{flood}=\rho_GU_{G,flood}")

        st.markdown("### 9.3 Capacity parameter")
        st.latex(r"C_S=U_{G,flood}\sqrt{\frac{\rho_G}{\rho_L-\rho_G}}")
        st.latex(r"C_P=C_S\sqrt{F_P}\,\nu_L^{0.05}")
        st.write("In this GPDC implementation, U is converted to ft/s, F_P is in ft⁻¹ and ν_L is in cSt.")

        st.markdown("### 9.4 Flood-curve numerical fit used by this code")
        st.latex(r"x=\log_{10}(F_{LV})")
        st.latex(r"C_{P,corr}=0.0394x^3+0.0552x^2-0.7634x+0.7863")
        st.write(
            "This polynomial is the **numerical curve fit implemented in the current application**. The solver changes "
            "U_G,flood until C_P calculated from the trial velocity equals the fitted flood-curve C_P. The code searches "
            "within F_LV = 0.01–8 and uses Brent's method for the final root."
        )

        st.markdown("### 9.5 Percent flood")
        st.latex(r"\%Flood=100\frac{U_{G,oper}}{U_{G,flood}}")
        st.write(
            "This is the reported flooding percentage. It is therefore a **velocity/capacity ratio**, not the square root "
            "of a pressure-drop ratio."
        )

        st.markdown("### 9.6 Kister–Gill flood-pressure-drop diagnostic")
        st.latex(r"\left(\frac{\Delta P}{Z}\right)_{flood}=0.115F_P^{0.7}\quad [in.H_2O/ft]")
        st.write(
            "This value is retained as a separate diagnostic cross-check. It does **not** determine %Flood in the current model."
        )

        hyd_walk = pd.DataFrame([
            ["Packing factor Fp", result["Fp_ft"], "ft⁻¹", result["Fp_basis"]],
            ["F_LV at solved flood", result["F_LV_flood"], "-", "GPDC"],
            ["CP at solved flood", result["CP_flood"], "-", "GPDC polynomial fit"],
            ["Operating gas velocity", result["Ug"], "m/s", "Q/A"],
            ["Solved flood velocity", result["U_flood"], "m/s", "GPDC root"],
            ["Percent flood", result["flooding_percent"], "%", "Ug/Uflood"],
            ["Kister–Gill flood ΔP", result["dP_flood_mbar_m"], "mbar/m", "diagnostic only"],
        ], columns=["Quantity", "Current value", "Unit", "Method"])
        st.dataframe(hyd_walk, use_container_width=True, hide_index=True)

    with st.expander("10. Automatic engineering diagnostics and warning thresholds"):
        st.markdown(
            """
The code currently generates warnings using the following **screening thresholds**:

- `Re_L < 10` → low liquid loading / incomplete wetting risk.
- `Re_G < 50` → gas-side Onda correlation is being used in a low-Reynolds region.
- `a_e/a_t < 0.30` → poor wetting / liquid-distribution risk.
- `A ≤ 1` → absorption is thermodynamically/operationally difficult for that component.
- No empirical `F_P` → geometric GPDC packing-factor fallback warning.
- GPDC root outside intended `F_LV` range → correlation-range warning.
- Flooding ≥ 80% → high hydraulic loading warning; ≥90% near-flood warning; ≥100% flooding predicted.
- Liquid mass flux < 1 kg/m²·s → minimum-wetting **screen only**; packing-specific minimum wetting rate must still be checked.

These warning boundaries are **model diagnostics, not universal design standards**.
            """
        )
        if result["warnings"]:
            st.write("Warnings for the current case:")
            for warning in result["warnings"]:
                st.warning(warning)
        else:
            st.success("The current case does not trigger any programmed diagnostic warning.")

    with st.expander("11. Complete assumption matrix — what the model assumes"):
        assumptions = pd.DataFrame([
            ["Steady state", "All flows, properties and compositions are time-independent.", "Start-up, shutdown and transient disturbances are not represented."],
            ["Counter-current contact", "Gas moves upward and liquid downward through the packing.", "ODE boundary conditions are built around this configuration."],
            ["Fresh water", "Top liquid inlet has x_ACN = x_VAc = 0.", "Recirculated or pre-loaded wash water must be added explicitly in a future version."],
            ["Isothermal column", "One temperature is used throughout the packed section.", "Heat of absorption, evaporation cooling and temperature profiles are neglected."],
            ["Constant pressure for mass transfer", "Operating P is constant in equilibrium and ODE equations.", "Calculated ΔP is reported hydraulically but is not fed back into local P(z)."],
            ["Ideal gas", "Carrier-gas molar density follows PV=nRT.", "Real-gas effects are neglected."],
            ["Air-like carrier gas", "MW_G = 0.029 kg/mol and Sutherland air viscosity are used.", "Different carrier-gas compositions require property-model revision."],
            ["Dilute VOC", "ACN/VAc do not materially change total gas or liquid flow/property values.", "Large solute loadings may violate this approximation."],
            ["Linear Henry equilibrium", "y* = mx with temperature-corrected H.", "Liquid activity coefficients and finite-concentration non-ideality are omitted."],
            ["Independent solutes", "ACN and VAc are solved separately.", "Cross-solute thermodynamic interactions are neglected."],
            ["No chemical reaction", "Absorption is physical only.", "Hydrolysis, polymerization or reactive scrubbing are not represented."],
            ["Pure-water bulk properties", "ρ, μ and σ are functions of T only.", "Dissolved VOC, salts or additives do not alter water properties."],
            ["Fixed diffusivities", "D_L and D_G are fixed component database values.", "Their dependence on T, P and composition is not modelled."],
            ["Onda applicability", "The same Onda framework is applied to all listed random packings using their stored geometry.", "Modern/high-performance packing may require vendor-specific mass-transfer correlations."],
            ["Uniform wetting/distribution", "a_e correlation represents average packed-bed wetting.", "Maldistribution, wall flow, distributor quality and local dry zones are not spatially resolved."],
            ["No axial dispersion", "Gas and liquid are treated as plug-flow phases in the axial direction.", "Backmixing/dispersion effects are neglected."],
            ["No entrainment/foaming", "Hydraulics do not explicitly model foam or droplet entrainment.", "Foaming or surface-active contamination can materially reduce capacity."],
            ["No fouling/deposition", "Packing geometry remains clean and constant.", "Polymer deposition or solids can change a_e, ΔP and flooding behavior."],
            ["Screening ΔP", "Pressure drop uses the code's holdup + geometry correction model.", "Vendor pressure-drop curves or plant data should supersede it for final rating."],
            ["GPDC flood capacity", "Flooding is estimated from a fitted GPDC flood line and packing factor.", "Accuracy depends strongly on correct empirical F_P and applicable packing family."],
            ["Ideal column internals", "Distributor, redistributor, support plate and demister losses are excluded.", "Total vessel ΔP will be larger than packed-bed ΔP alone."],
        ], columns=["Area", "Assumption used", "Practical consequence / limitation"])
        st.dataframe(assumptions, use_container_width=True, hide_index=True)

    with st.expander("12. Packing database used by the application"):
        packing_rows = []
        for name, p in PACKING_DATA.items():
            packing_rows.append({
                "Packing": name,
                "a_t (m²/m³)": p["a"],
                "d_p (m)": p["d"],
                "Void fraction ε": p["epsilon"],
                "Critical surface tension σc (N/m)": p["sigma_c"],
                "Pressure-drop ψ": p["psi"],
                "Fp (ft⁻¹)": p["Fp_ft"] if p["Fp_ft"] is not None else "fallback",
                "Fp basis": p["Fp_basis"],
            })
        st.dataframe(pd.DataFrame(packing_rows), use_container_width=True, hide_index=True)
        st.warning(
            "Only packings with a stored empirical Fp use that value directly. Entries marked **geometric_fallback** "
            "estimate Fp from a_t/ε³ and should be treated with lower confidence. The current database does not embed "
            "vendor certificates/references for every a_t, ε, ψ and Fp value; those should be replaced with verified vendor data where available."
        )

    with st.expander("13. Model parameters that still require independent source / plant verification"):
        verify_df = pd.DataFrame([
            ["ACN Henry H25 and temperature coefficient", "Directly controls equilibrium slope m and therefore A, driving force and outlet.", "High"],
            ["VAc Henry H25 and temperature coefficient", "VAc is often the limiting component in this model.", "High"],
            ["Liquid and gas diffusivities D_L, D_G", "Enter Onda Sc numbers and kL/kG.", "Medium–High"],
            ["Packing a_t, ε, d_p, σ_c", "Control wetting and transfer coefficients.", "High"],
            ["Empirical packing factor Fp", "Directly controls GPDC capacity and Kister–Gill flood ΔP.", "High"],
            ["Pressure-drop ψ multiplier", "Directly scales the screening dry pressure-drop equation.", "High for ΔP"],
            ["Fresh-water inlet assumption", "Real recirculated water may already contain ACN/VAc.", "High if recycle exists"],
            ["Plant inlet/outlet concentration basis", "mgC, mgVOC and ppmv are not interchangeable.", "Critical"],
            ["Actual vs normal gas-flow basis", "Changes gas velocity, residence time and flooding fraction.", "Critical"],
        ], columns=["Parameter / assumption", "Why it matters", "Priority"])
        st.dataframe(verify_df, use_container_width=True, hide_index=True)

    with st.expander("14. Correlation and reference map"):
        st.markdown(
            """
The application uses the following literature/standard correlation families. These links document the **method family**; the exact numerical database values stored for ACN, VAc and individual packings must still be verified separately.

- **Onda, Takeuchi & Okumoto (1968)** — wetted area and individual gas/liquid mass-transfer correlations for packed columns:  
  https://doi.org/10.1252/jcej.1.56
- **Kister et al. (2007), AIChE CEP — GPDC** — capacity parameter, C-factor, flow parameter and packing-factor framework:  
  https://publications.aiche.org/cep/2007/july/realistically-predict-capacity-and-pressure-drop-packed-columns
- **Kister–Gill flood-pressure-drop relation** — implemented as ΔP_flood = 0.115 Fp^0.7 in English units and used here only as a diagnostic.
- **IAPWS R1-76(2014)** — surface tension of ordinary water:  
  https://iapws.org/relguide/Surf-H2O.html
- **Water-density temperature fit** — the implemented equation is the commonly used 0–100 °C pure-water fit also documented by ORNL/MEASUR:  
  https://industrialresources.ornl.gov/measur/suite/docs/group__water__density__formula
- **Sutherland gas-viscosity law** — air viscosity temperature correction with μ₀ ≈ 1.716×10⁻⁵ Pa·s, T₀ ≈ 273 K and S ≈ 110–111 K.

**Important:** the current pressure-drop equation and GPDC polynomial are implementation-level engineering approximations/curve fits in this application. They should be validated against vendor hydraulic data before final equipment design.
            """
        )

    with st.expander("15. What the model does NOT yet calculate"):
        st.markdown(
            """
The present version does **not** explicitly calculate:

- heat of absorption or axial temperature change,
- water evaporation and humidity balance,
- ACN/VAc liquid-phase activity coefficients (NRTL/UNIQUAC, etc.),
- concentration-dependent Henry constants,
- chemical reaction, polymerization or degradation,
- axial dispersion/backmixing,
- distributor/redistributor hydraulics,
- support-plate and demister pressure drop,
- droplet entrainment, foaming or aerosol formation,
- packing fouling / polymer deposition,
- wall flow or detailed maldistribution,
- dynamic start-up/shutdown behavior,
- recirculating-solvent inventory and accumulation,
- pump, tank, cooler or full-loop energy balances,
- vendor-specific rated capacity/performance curves.

For this reason the application should be treated as a **mechanistic engineering screening and design-support model**, not as a certified vendor rating package.
            """
        )

st.caption("Engineering screening / design-support tool — validate against plant and vendor data before final design decisions.")
