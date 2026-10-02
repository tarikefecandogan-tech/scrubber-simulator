import math
from dataclasses import asdict, dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scrubber_model import ScrubberSimulationV3, PACKING_DATA


st.set_page_config(
    page_title="Scrubber Design Simulator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# SMALL HELPERS
# -----------------------------------------------------------------------------
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
    inlet_toc_mg_Nm3: float
    acn_mass_fraction: float
    vac_mass_fraction: float
    target_toc_mg_Nm3: float


def make_sim(i: Inputs) -> ScrubberSimulationV3:
    return ScrubberSimulationV3(
        Dolgu_Tipi=i.packing,
        Kolon_Capi_D=i.diameter_m,
        Dolgu_Yuksekligi_Z=i.packed_height_m,
        Sivi_Debisi_L=i.liquid_kg_h,
        Gaz_Debisi_Q=i.gas_m3_h,
        Debi_Referansi=i.flow_reference,
        Sicaklik_T=i.temperature_C,
        P_operating=i.pressure_Pa,
        Giris_TOC_N=i.inlet_toc_mg_Nm3,
        ACN_Kutle_Kesri=i.acn_mass_fraction,
        VAc_Kutle_Kesri=i.vac_mass_fraction,
    )


def fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "—"
    return f"{value:.{digits}f}"


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
        return "🟢 LOW–MODERATE LOADING"
    return "🔵 UNDERLOADED / HIGH CAPACITY MARGIN"


def result_row(i: Inputs, result: Dict[str, Any], z_req):
    return {
        "Packing": i.packing,
        "Diameter_m": i.diameter_m,
        "Packed_height_m": i.packed_height_m,
        "Liquid_kg_h": i.liquid_kg_h,
        "Gas_m3_h_input": i.gas_m3_h,
        "Flow_reference": i.flow_reference,
        "Temperature_C": i.temperature_C,
        "Pressure_bar_abs": i.pressure_Pa / 1e5,
        "Inlet_TOC_mg_Nm3": i.inlet_toc_mg_Nm3,
        "Outlet_TOC_mg_Nm3": result["outlet_TOC_N"],
        "Removal_percent": 100 * result["overall_removal"],
        "ACN_removal_percent": 100 * result["ACN_removal"],
        "VAc_removal_percent": 100 * result["VAc_removal"],
        "Required_height_m": z_req,
        "Ug_m_s": result["Ug"],
        "U_flood_m_s": result["U_flood"],
        "Flood_percent": result["flooding_percent"],
        "dP_mbar_m": result["dPwet_mbar_m"],
        "Total_dP_mbar": result["Total_dP_mbar"],
        "F_oper_Pa05": result["F_oper"],
        "F_flood_Pa05": result["F_flood"],
        "F_LV_flood": result["F_LV_flood"],
        "Packing_factor_ft-1": result["Fp_ft"],
        "Packing_factor_basis": result["Fp_basis"],
        "Hydraulic_regime": result["hydraulic_regime"],
    }


@st.cache_data(show_spinner=False)
def run_cached(data: dict, profiles: bool = True):
    i = Inputs(**data)
    sim = make_sim(i)
    result = sim.simulate(return_profiles=profiles)
    z_req = sim.required_height(i.target_toc_mg_Nm3)
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
            z_req = sim.required_height(i.target_toc_mg_Nm3)
            rows.append(result_row(i, r, z_req))
        except Exception as exc:
            rows.append({"Packing": packing, "Error": str(exc)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def sweep_cached(data: dict, parameter: str, start: float, stop: float, points: int):
    base = Inputs(**data)
    values = np.linspace(start, stop, points)
    rows = []

    param_map = {
        "Liquid flow (kg/h)": "liquid_kg_h",
        "Gas flow (m³/h)": "gas_m3_h",
        "Packed height (m)": "packed_height_m",
        "Column diameter (m)": "diameter_m",
        "Temperature (°C)": "temperature_C",
    }
    field = param_map[parameter]

    for v in values:
        try:
            i = Inputs(**{**asdict(base), field: float(v)})
            sim = make_sim(i)
            r = sim.simulate(return_profiles=False)
            rows.append({
                "x": float(v),
                "Outlet TOC (mg/Nm³)": r["outlet_TOC_N"],
                "Removal (%)": 100 * r["overall_removal"],
                "Flooding (%)": r["flooding_percent"],
                "ΔP (mbar/m)": r["dPwet_mbar_m"],
            })
        except Exception:
            rows.append({
                "x": float(v),
                "Outlet TOC (mg/Nm³)": np.nan,
                "Removal (%)": np.nan,
                "Flooding (%)": np.nan,
                "ΔP (mbar/m)": np.nan,
            })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# SIDEBAR — INPUT FORM
# -----------------------------------------------------------------------------
st.title("🧪 Scrubber Design Simulator")
st.caption(
    "ACN + Vinyl Acetate absorption into water | ODE mass transfer + GPDC flooding capacity"
)

with st.sidebar:
    st.header("Model Inputs")
    st.caption("Values are only recalculated when you press **Run simulation**.")

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
        temp = st.number_input("Temperature (°C)", min_value=-20.0, max_value=100.0, value=22.0, step=1.0)
        pressure_bar = st.number_input("Operating pressure (bar abs)", min_value=0.10, value=1.01325, step=0.05, format="%.5f")

        st.subheader("Feed & Target")
        toc_in = st.number_input("Inlet concentration (mg/Nm³)", min_value=0.001, value=5000.0, step=100.0)
        acn_pct = st.slider("ACN mass fraction (%)", min_value=0.0, max_value=100.0, value=93.0, step=0.5)
        vac_pct = 100.0 - acn_pct
        st.caption(f"VAc mass fraction is automatically set to **{vac_pct:.1f}%**.")
        target = st.number_input("Target outlet (mg/Nm³)", min_value=0.0001, value=20.0, step=1.0)

        submitted = st.form_submit_button("▶ Run simulation", use_container_width=True, type="primary")

    st.divider()
    st.info(
        "The model currently interprets the concentration input as **mg of VOC per Nm³**. "
        "If the plant instrument reports mgC/Nm³ or ppmv, convert the basis before using the result quantitatively."
    )


inputs = Inputs(
    packing=packing,
    diameter_m=float(diameter),
    packed_height_m=float(height),
    liquid_kg_h=float(liquid),
    gas_m3_h=float(gas),
    flow_reference=flow_ref,
    temperature_C=float(temp),
    pressure_Pa=float(pressure_bar) * 1e5,
    inlet_toc_mg_Nm3=float(toc_in),
    acn_mass_fraction=float(acn_pct) / 100.0,
    vac_mass_fraction=float(vac_pct) / 100.0,
    target_toc_mg_Nm3=float(target),
)

# Persist latest successful input set. On first load, defaults are run automatically.
if "active_inputs" not in st.session_state:
    st.session_state.active_inputs = asdict(inputs)
if submitted:
    st.session_state.active_inputs = asdict(inputs)

active = Inputs(**st.session_state.active_inputs)

try:
    with st.spinner("Running scrubber model..."):
        result, z_req = run_cached(asdict(active), profiles=True)
except Exception as exc:
    st.error("Simulation could not be completed.")
    st.exception(exc)
    st.stop()


# -----------------------------------------------------------------------------
# TOP SUMMARY
# -----------------------------------------------------------------------------
metric_cols = st.columns(5)
metric_cols[0].metric("Outlet TOC", f"{result['outlet_TOC_N']:.2f} mg/Nm³")
metric_cols[1].metric("Overall removal", f"{100*result['overall_removal']:.2f} %")
metric_cols[2].metric("Required height", "Not reached" if z_req is None else f"{z_req:.3f} m")
metric_cols[3].metric("Flooding", f"{result['flooding_percent']:.1f} %")
metric_cols[4].metric("Pressure drop", f"{result['dPwet_mbar_m']:.3f} mbar/m")

if result["outlet_TOC_N"] <= active.target_toc_mg_Nm3:
    st.success(f"Target achieved: outlet ≤ {active.target_toc_mg_Nm3:g} mg/Nm³")
else:
    st.warning(f"Target not achieved at the current packed height of {active.packed_height_m:g} m.")


# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab_overview, tab_mt, tab_hyd, tab_profiles, tab_compare, tab_sweep, tab_method = st.tabs([
    "Overview",
    "Mass Transfer",
    "Hydraulics",
    "Profiles",
    "Packing Comparison",
    "Parameter Sweep",
    "Method & Assumptions",
])


with tab_overview:
    left, right = st.columns([1.15, 0.85])

    with left:
        st.subheader("Performance summary")
        perf = pd.DataFrame([
            ["Total", active.inlet_toc_mg_Nm3, result["outlet_TOC_N"], 100*result["overall_removal"]],
            ["ACN", result["ACN_in"], result["ACN_out"], 100*result["ACN_removal"]],
            ["VAc", result["VAc_in"], result["VAc_out"], 100*result["VAc_removal"]],
        ], columns=["Component", "Inlet (mg/Nm³)", "Outlet (mg/Nm³)", "Removal (%)"])
        st.dataframe(perf, use_container_width=True, hide_index=True)

        st.subheader("Selected design")
        design_df = pd.DataFrame({
            "Parameter": [
                "Packing", "Diameter", "Packed height", "Water flow", "Gas flow input",
                "Flow reference", "Temperature", "Pressure", "Target outlet"
            ],
            "Value": [
                active.packing, f"{active.diameter_m:g} m", f"{active.packed_height_m:g} m",
                f"{active.liquid_kg_h:g} kg/h", f"{active.gas_m3_h:g} m³/h",
                active.flow_reference, f"{active.temperature_C:g} °C",
                f"{active.pressure_Pa/1e5:.5g} bar abs", f"{active.target_toc_mg_Nm3:g} mg/Nm³"
            ]
        })
        st.dataframe(design_df, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Hydraulic status")
        st.markdown(f"### {hydraulic_badge(result['flooding_percent'])}")
        st.progress(min(max(result["flooding_percent"] / 100.0, 0.0), 1.0))
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
        file_name="scrubber_result.csv",
        mime="text/csv",
    )


with tab_mt:
    st.subheader("Mass-transfer coefficients")
    rows = []
    for comp in ["ACN", "VAc"]:
        c = result[comp]["coefficients"]
        rows.append({
            "Component": comp,
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
        })
    mt_df = pd.DataFrame(rows)
    st.dataframe(mt_df, use_container_width=True, hide_index=True)

    st.caption(
        "The model calculates ACN and VAc independently, including component-specific Henry constants and diffusivities."
    )


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
            f"{result['dP_flood_mbar_m']:.3f} mbar/m",
            f"{result['dP_ratio_to_flood']:.5f}",
        ]
    })
    st.dataframe(hydraulic_df, use_container_width=True, hide_index=True)

    st.info(
        "Flooding percentage is based on the independent GPDC flood velocity: "
        "%Flood = Ug,oper / Ug,flood × 100. The Kister–Gill flood pressure-drop value is shown only as a diagnostic."
    )


with tab_profiles:
    st.subheader("Gas-phase concentration through the packed height")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    profile_sim = make_sim(active)
    for comp in ["ACN", "VAc"]:
        z = np.asarray(result[comp]["z"])
        y = np.asarray(result[comp]["y_profile"])
        toc = np.array([profile_sim.y_to_toc(val, comp) for val in y])
        ax.plot(z, toc, marker="o", markersize=2.5, label=comp)
    ax.set_xlabel("Packed height coordinate, z (m)")
    ax.set_ylabel("Gas concentration (mg/Nm³)")
    ax.set_title("Gas-phase concentration profile")
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.caption(
        "The model coordinate starts at the gas inlet / liquid outlet end of the counter-current packed section."
    )


with tab_compare:
    st.subheader("Compare packings at the same operating conditions")
    selected = st.multiselect(
        "Packings",
        list(PACKING_DATA.keys()),
        default=["25mm Metal Pall Ring", "38mm Metal Pall Ring", "IMTP #25 (Metal)"]
    )
    if selected:
        comp_df = packing_comparison_cached(asdict(active), tuple(selected))
        display_cols = [c for c in [
            "Packing", "Outlet_TOC_mg_Nm3", "Removal_percent", "Required_height_m",
            "Flood_percent", "dP_mbar_m", "Packing_factor_ft-1", "Packing_factor_basis", "Error"
        ] if c in comp_df.columns]
        st.dataframe(comp_df[display_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download packing comparison",
            comp_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="packing_comparison.csv",
            mime="text/csv",
        )
    else:
        st.info("Select at least one packing.")


with tab_sweep:
    st.subheader("Sensitivity / parameter sweep")
    col_a, col_b, col_c, col_d = st.columns(4)
    sweep_parameter = col_a.selectbox(
        "Parameter",
        ["Liquid flow (kg/h)", "Gas flow (m³/h)", "Packed height (m)", "Column diameter (m)", "Temperature (°C)"]
    )

    defaults = {
        "Liquid flow (kg/h)": (max(100.0, active.liquid_kg_h*0.4), active.liquid_kg_h*1.6),
        "Gas flow (m³/h)": (max(1.0, active.gas_m3_h*0.4), active.gas_m3_h*2.0),
        "Packed height (m)": (max(0.1, active.packed_height_m*0.4), active.packed_height_m*2.0),
        "Column diameter (m)": (max(0.1, active.diameter_m*0.6), active.diameter_m*1.5),
        "Temperature (°C)": (max(-10.0, active.temperature_C-10.0), min(80.0, active.temperature_C+20.0)),
    }
    d_start, d_stop = defaults[sweep_parameter]
    start = col_b.number_input("Start", value=float(d_start))
    stop = col_c.number_input("Stop", value=float(d_stop))
    npts = col_d.slider("Points", min_value=5, max_value=50, value=16)

    if stop <= start:
        st.error("Stop must be greater than Start.")
    else:
        sweep_df = sweep_cached(asdict(active), sweep_parameter, float(start), float(stop), int(npts))
        renamed = sweep_df.rename(columns={"x": sweep_parameter})
        st.dataframe(renamed, use_container_width=True, hide_index=True)

        plot1 = sweep_df.set_index("x")[["Outlet TOC (mg/Nm³)"]]
        st.line_chart(plot1, x_label=sweep_parameter, y_label="Outlet TOC (mg/Nm³)")

        plot2 = sweep_df.set_index("x")[["Flooding (%)", "Removal (%)"]]
        st.line_chart(plot2, x_label=sweep_parameter, y_label="Percent")

        st.download_button(
            "⬇ Download sweep data",
            renamed.to_csv(index=False).encode("utf-8-sig"),
            file_name="parameter_sweep.csv",
            mime="text/csv",
        )


with tab_method:
    st.subheader("What the model currently does")
    st.markdown(
        """
1. Converts the entered gas flow to actual flow when the input reference is normal conditions.
2. Treats ACN and VAc as separate dilute solutes in water.
3. Calculates temperature-dependent Henry constants.
4. Estimates wetted packing area and individual gas/liquid mass-transfer coefficients.
5. Combines film resistances into an overall gas-side coefficient, **K_G**.
6. Solves the counter-current gas/liquid concentration profiles with an ODE shooting method.
7. Calculates packed-bed pressure drop separately from flooding capacity.
8. Calculates GPDC flood velocity independently and reports **%Flood = U_oper/U_flood × 100**.
9. Solves the packed height required to meet the selected target concentration.
        """
    )

    st.subheader("Important current assumptions")
    st.markdown(
        """
- Concentration basis is currently **mg VOC/Nm³**, not automatically mgC/Nm³ or ppmv.
- Water properties are currently fixed rather than temperature-dependent.
- Gas is represented with air-like bulk properties.
- ACN and VAc are solved independently; multicomponent liquid non-ideality is not included.
- Some packing types have empirical GPDC packing factors; others use a geometric fallback and trigger a warning.
- The pressure-drop model is a screening-level engineering model, not a vendor rating program.
- Quantitative plant prediction still requires calibration/validation against measured scrubber data.
        """
    )

    st.subheader("Core flooding definition")
    st.latex(r"F_{LV}=\frac{L'}{G'_{flood}}\sqrt{\frac{\rho_G}{\rho_L}}")
    st.latex(r"C_S=U_{G,flood}\sqrt{\frac{\rho_G}{\rho_L-\rho_G}}")
    st.latex(r"\%Flood=100\frac{U_{G,oper}}{U_{G,flood}}")

st.caption("Engineering screening / design-support tool — validate against plant and vendor data before final design decisions.")
