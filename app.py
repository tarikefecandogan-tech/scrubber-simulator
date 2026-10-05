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
    page_title="Scrubber Design Simulator V3.4",
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


st.title("🧪 Scrubber Design Simulator — V3.4")
st.caption(
    "ACN + Vinyl Acetate absorption into water | ODE mass transfer + GPDC flooding | "
    "A, HTU/NTU, temperature-dependent water properties and concentration-basis conversion"
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
    st.subheader("V3.4 additions")
    st.markdown(
        """
- **Absorption factor:** component-specific `A = L/(mG)`.
- **HTU/NTU diagnostics:** `HTU_OG = G/(K_G a_e P)` and `NTU_OG = Z/HTU_OG`.
- **Temperature-dependent water:** density, dynamic viscosity and surface tension vary with selected temperature.
- **Concentration basis:** inlet, target and outlet can be handled directly as **mgVOC/Nm³**, **mgC/Nm³** or **ppmv**.
        """
    )
    st.subheader("Important assumptions")
    st.markdown(
        """
- ACN/VAc composition is entered as **VOC mass fraction**, even when the concentration basis is ppmv.
- The ppmv conversion therefore converts that mass composition to the corresponding VOC molar composition.
- Normal concentration basis uses **0 °C and 1 atm**.
- Gas bulk properties are air-like and the solutes are dilute.
- ACN and VAc are solved independently; multicomponent liquid non-ideality is not included.
- Liquid diffusivities are still fixed reference values; only bulk water ρ, μ and σ are temperature-dependent in this version.
- Pressure drop is a screening-level model; final design should be checked against vendor/plant data.
        """
    )
    st.latex(r"A_i=\frac{L}{m_iG}")
    st.latex(r"HTU_{OG,i}=\frac{G'}{K_{G,i}a_{e,i}P}")
    st.latex(r"NTU_{OG,i}=\frac{Z}{HTU_{OG,i}}")
    st.latex(r"\%Flood=100\frac{U_{G,oper}}{U_{G,flood}}")

st.caption("Engineering screening / design-support tool — validate against plant and vendor data before final design decisions.")
