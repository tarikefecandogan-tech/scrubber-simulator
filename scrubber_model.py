"""
Scrubber Simulation V3.4 — engineering calculation engine.

New in V3.4
-----------
1) Component absorption factor A = L/(mG)
2) Overall gas-side HTU/NTU diagnostics
3) Temperature-dependent water density, viscosity and surface tension
4) Concentration basis support: mgVOC/Nm3, mgC/Nm3 and ppmv

Core model: ACN/VAc absorption into water, ODE-based counter-current mass transfer,
separate screening pressure-drop model and GPDC-based flooding capacity estimate.
"""

import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ============================================================
# DEFAULTS
# ============================================================
Dolgu_Tipi = '25mm Metal Pall Ring'
Kolon_Capi_D = 0.5
Dolgu_Yuksekligi_Z = 1.4
Sivi_Debisi_L = 2500.0
Gaz_Debisi_Q = 117.53
Debi_Referansi = 'Aktual'
Sicaklik_T = 22.0
P_operating = 101325.0
Giris_Konsantrasyonu = 5000.0
Konsantrasyon_Bazi = 'mgVOC/Nm³'
ACN_Kutle_Kesri = 0.93
VAc_Kutle_Kesri = 0.07
Hedef_Outlet = 20.0

# ============================================================
# CONSTANTS / DATABASES
# ============================================================
T_N = 273.15
P_N = 101325.0
R = 8.314462618
g = 9.81
FT_PER_M = 3.280839895
PA_PER_INH2O = 249.08891
M_PER_FT = 0.3048
MBAR_PER_PA = 0.01
MW_C = 12.011e-3  # kg/mol carbon atom
WATER_MW = 18.015e-3

CONCENTRATION_BASES = ('mgVOC/Nm³', 'mgC/Nm³', 'ppmv')

GAS = {
    'MW': 0.029,
    'mu_ref': 1.716e-5,
    'T_ref': 273.15,
    'S': 110.4,
}

VOC = {
    'ACN': {
        'MW': 53.06e-3,
        'nC': 3,
        'H25_atm_m3_mol': 1.18e-5,
        'DH_H_over_R': 4200.0,
        'DL': 1.0e-9,
        'DG': 1.0e-5,
    },
    'VAc': {
        'MW': 86.09e-3,
        'nC': 4,
        'H25_atm_m3_mol': 5.11e-4,
        'DH_H_over_R': 4500.0,
        'DL': 0.9e-9,
        'DG': 0.9e-5,
    },
}

PACKING_DATA = {
    '25mm Metal Pall Ring': {
        'a': 212.0, 'd': 0.025, 'epsilon': 0.962,
        'sigma_c': 0.075, 'psi': 1.20,
        'Fp_ft': 48.0, 'Fp_basis': 'empirical'
    },
    '38mm Metal Pall Ring': {
        'a': 145.0, 'd': 0.038, 'epsilon': 0.967,
        'sigma_c': 0.075, 'psi': 1.00,
        'Fp_ft': 28.0, 'Fp_basis': 'empirical'
    },
    'IMTP #25 (Metal)': {
        'a': 226.0, 'd': 0.025, 'epsilon': 0.970,
        'sigma_c': 0.075, 'psi': 1.10,
        'Fp_ft': 41.0, 'Fp_basis': 'empirical'
    },
    'IMTP #40 (Metal)': {
        'a': 150.0, 'd': 0.040, 'epsilon': 0.975,
        'sigma_c': 0.075, 'psi': 0.90,
        'Fp_ft': 24.0, 'Fp_basis': 'empirical'
    },
    'CMR #2 (Metal)': {
        'a': 188.0, 'd': 0.025, 'epsilon': 0.967,
        'sigma_c': 0.075, 'psi': 1.00,
        'Fp_ft': None, 'Fp_basis': 'geometric_fallback'
    },
    'CMR #3 (Metal)': {
        'a': 135.0, 'd': 0.040, 'epsilon': 0.972,
        'sigma_c': 0.075, 'psi': 0.80,
        'Fp_ft': None, 'Fp_basis': 'geometric_fallback'
    },
    '13mm Super Raschig (MSRT-02)': {
        'a': 250.0, 'd': 0.013, 'epsilon': 0.965,
        'sigma_c': 0.075, 'psi': 1.0,
        'Fp_ft': None, 'Fp_basis': 'geometric_fallback'
    },
    '16mm Super Raschig (MSRT-03)': {
        'a': 215.0, 'd': 0.016, 'epsilon': 0.961,
        'sigma_c': 0.075, 'psi': 0.9,
        'Fp_ft': None, 'Fp_basis': 'geometric_fallback'
    },
    '25mm Super Raschig (MSRT-05)': {
        'a': 150.0, 'd': 0.025, 'epsilon': 0.972,
        'sigma_c': 0.075, 'psi': 0.8,
        'Fp_ft': None, 'Fp_basis': 'geometric_fallback'
    },
    '38mm Super Raschig (MSRT-06)': {
        'a': 120.0, 'd': 0.038, 'epsilon': 0.978,
        'sigma_c': 0.075, 'psi': 0.7,
        'Fp_ft': None, 'Fp_basis': 'geometric_fallback'
    },
}


def air_viscosity(T_kelvin):
    return (
        GAS['mu_ref']
        * (T_kelvin / GAS['T_ref']) ** 1.5
        * (GAS['T_ref'] + GAS['S'])
        / (T_kelvin + GAS['S'])
    )


def gas_density(P, T):
    return P * GAS['MW'] / (R * T)


def henry_constant_Pa_m3_mol(T_kelvin, component):
    H25 = VOC[component]['H25_atm_m3_mol']
    DH_over_R = VOC[component]['DH_H_over_R']
    H_atm = H25 * math.exp(DH_over_R * (1.0 / 298.15 - 1.0 / T_kelvin))
    return H_atm * P_N


def water_properties(T_C):
    """Approximate pure-water properties as functions of temperature.

    Returns rho [kg/m3], mu [Pa.s], sigma [N/m].
    Density uses a common 0-100 C freshwater fit; viscosity uses an Andrade-type
    expression; surface tension uses the IAPWS-style reduced-temperature form.
    The scrubber UI limits normal use to the liquid-water range.
    """
    if not (-10.0 <= T_C <= 100.0):
        raise ValueError("Water-property correlation is intended for approximately -10 to 100 °C.")

    T_K = T_C + 273.15

    # Density: commonly used Kell-type correlation near atmospheric pressure.
    rho = 1000.0 * (
        1.0
        - ((T_C + 288.9414) / (508929.2 * (T_C + 68.12963)))
        * (T_C - 3.9863) ** 2
    )

    # Dynamic viscosity [Pa.s].
    mu = 2.414e-5 * 10.0 ** (247.8 / (T_K - 140.0))

    # Surface tension [N/m], IAPWS-style correlation.
    Tc = 647.096
    tau = max(1e-12, 1.0 - T_K / Tc)
    sigma = 0.2358 * tau ** 1.256 * (1.0 - 0.625 * tau)

    return {'rho': rho, 'mu': mu, 'sigma': sigma, 'MW': WATER_MW}


class ScrubberSimulationV34:
    def __init__(
        self,
        Dolgu_Tipi=Dolgu_Tipi,
        Kolon_Capi_D=Kolon_Capi_D,
        Dolgu_Yuksekligi_Z=Dolgu_Yuksekligi_Z,
        Sivi_Debisi_L=Sivi_Debisi_L,
        Gaz_Debisi_Q=Gaz_Debisi_Q,
        Debi_Referansi=Debi_Referansi,
        Sicaklik_T=Sicaklik_T,
        P_operating=P_operating,
        Giris_TOC_N=Giris_Konsantrasyonu,
        ACN_Kutle_Kesri=ACN_Kutle_Kesri,
        VAc_Kutle_Kesri=VAc_Kutle_Kesri,
        Konsantrasyon_Bazi=Konsantrasyon_Bazi,
    ):
        self.Dolgu_Tipi = Dolgu_Tipi
        self.D = float(Kolon_Capi_D)
        self.Z = float(Dolgu_Yuksekligi_Z)
        self.L = float(Sivi_Debisi_L)
        self.Q_input = float(Gaz_Debisi_Q)
        self.Debi_Referansi = Debi_Referansi
        self.T_C = float(Sicaklik_T)
        self.T = self.T_C + 273.15
        self.P = float(P_operating)
        self.input_concentration = float(Giris_TOC_N)
        self.concentration_basis = Konsantrasyon_Bazi
        self.w_ACN = float(ACN_Kutle_Kesri)
        self.w_VAc = float(VAc_Kutle_Kesri)

        if self.Dolgu_Tipi not in PACKING_DATA:
            raise ValueError(f"Unknown packing: {self.Dolgu_Tipi}")
        if self.concentration_basis not in CONCENTRATION_BASES:
            raise ValueError(f"Unknown concentration basis: {self.concentration_basis}")
        if abs(self.w_ACN + self.w_VAc - 1.0) > 1e-8:
            raise ValueError("ACN + VAc mass fractions must sum to 1.0.")
        if self.D <= 0 or self.Z <= 0 or self.L <= 0 or self.Q_input <= 0:
            raise ValueError("All physical inputs must be greater than zero.")
        if self.P <= 0 or self.input_concentration < 0:
            raise ValueError("Pressure must be positive and concentration non-negative.")

        self.A = math.pi * self.D**2 / 4.0
        self.rho_G = gas_density(self.P, self.T)
        self.mu_G = air_viscosity(self.T)
        self.water = water_properties(self.T_C)

        if self.Debi_Referansi.lower().startswith('normal'):
            self.Q = self.Q_input * (self.T / T_N) * (P_N / self.P)
        else:
            self.Q = self.Q_input

        # Convert the entered total concentration to component inlet mole fractions.
        self.y_in_components = self._total_input_to_component_y(
            self.input_concentration, self.concentration_basis
        )

        # Canonical inlet values in all supported bases.
        self.inlet_by_basis = self.total_concentrations_from_y(self.y_in_components)

        # Backward-compatible canonical mgVOC variable.
        self.TOC_in = self.inlet_by_basis['mgVOC/Nm³']

    # --------------------------------------------------------
    # FLOW HELPERS
    # --------------------------------------------------------
    def gas_molar_flow(self):
        return self.Q * self.P / (R * self.T) / 3600.0

    def liquid_molar_flow(self):
        return self.L / WATER_MW / 3600.0

    def gas_mass_flux(self):
        return self.Q * self.rho_G / 3600.0 / self.A

    def liquid_mass_flux(self):
        return self.L / 3600.0 / self.A

    def gas_superficial_velocity(self):
        return self.Q / 3600.0 / self.A

    # --------------------------------------------------------
    # CONCENTRATION BASIS CONVERSIONS
    # --------------------------------------------------------
    @staticmethod
    def normal_gas_molar_concentration():
        return P_N / (R * T_N)  # mol/Nm3

    @staticmethod
    def carbon_mass_fraction(component):
        v = VOC[component]
        return v['nC'] * MW_C / v['MW']

    def _mass_fraction(self, component):
        return self.w_ACN if component == 'ACN' else self.w_VAc

    def _total_input_to_component_y(self, total_value, basis):
        """Convert total mixture concentration to component gas mole fractions.

        ACN/VAc composition is entered as VOC mass fraction. For ppmv input the
        mass fractions are converted to the corresponding VOC molar fractions.
        """
        Cgas_N = self.normal_gas_molar_concentration()
        weights = {'ACN': self.w_ACN, 'VAc': self.w_VAc}

        if basis == 'mgVOC/Nm³':
            total_voc_mg = total_value
            y = {}
            for comp in ('ACN', 'VAc'):
                comp_kg_Nm3 = total_voc_mg * weights[comp] * 1e-6
                y[comp] = (comp_kg_Nm3 / VOC[comp]['MW']) / Cgas_N
            return y

        if basis == 'mgC/Nm³':
            mix_carbon_fraction = sum(
                weights[c] * self.carbon_mass_fraction(c) for c in ('ACN', 'VAc')
            )
            if mix_carbon_fraction <= 0:
                raise ValueError("Mixture carbon fraction is invalid.")
            total_voc_mg = total_value / mix_carbon_fraction
            y = {}
            for comp in ('ACN', 'VAc'):
                comp_kg_Nm3 = total_voc_mg * weights[comp] * 1e-6
                y[comp] = (comp_kg_Nm3 / VOC[comp]['MW']) / Cgas_N
            return y

        if basis == 'ppmv':
            denom = sum(weights[c] / VOC[c]['MW'] for c in ('ACN', 'VAc'))
            mole_share = {
                c: (weights[c] / VOC[c]['MW']) / denom for c in ('ACN', 'VAc')
            }
            return {c: total_value * 1e-6 * mole_share[c] for c in ('ACN', 'VAc')}

        raise ValueError(f"Unsupported basis: {basis}")

    def y_to_component_concentration(self, y, component, basis):
        Cgas_N = self.normal_gas_molar_concentration()
        if basis == 'mgVOC/Nm³':
            return y * Cgas_N * VOC[component]['MW'] * 1e6
        if basis == 'mgC/Nm³':
            return y * Cgas_N * VOC[component]['nC'] * MW_C * 1e6
        if basis == 'ppmv':
            return y * 1e6
        raise ValueError(f"Unsupported basis: {basis}")

    def total_concentrations_from_y(self, y_components):
        return {
            basis: sum(
                self.y_to_component_concentration(y_components[c], c, basis)
                for c in ('ACN', 'VAc')
            )
            for basis in CONCENTRATION_BASES
        }

    # Backward-compatible helpers for existing UI/profile code.
    def toc_to_y(self, TOC_N, component, mass_fraction):
        MW = VOC[component]['MW']
        C_mass_kg_Nm3 = TOC_N * mass_fraction * 1e-6
        C_molar_Nm3 = C_mass_kg_Nm3 / MW
        return C_molar_Nm3 / self.normal_gas_molar_concentration()

    def y_to_toc(self, y, component):
        return self.y_to_component_concentration(y, component, 'mgVOC/Nm³')

    # --------------------------------------------------------
    # MASS TRANSFER
    # --------------------------------------------------------
    def coefficients(self, component):
        p = PACKING_DATA[self.Dolgu_Tipi]
        a_t, dp, sigma_c = p['a'], p['d'], p['sigma_c']
        rho_L = self.water['rho']
        mu_L = self.water['mu']
        sigma_L = self.water['sigma']
        DL, DG = VOC[component]['DL'], VOC[component]['DG']

        Lp = self.liquid_mass_flux()
        Gp = self.gas_mass_flux()
        Re_L = Lp / (a_t * mu_L)
        Re_G = Gp / (a_t * self.mu_G)
        Fr_L = Lp**2 * a_t / (rho_L**2 * g)
        We_L = Lp**2 / (rho_L * sigma_L * a_t)

        wet_exp = (
            -1.45
            * (sigma_c / sigma_L)**0.75
            * max(Re_L, 1e-30)**0.10
            * max(Fr_L, 1e-30)**(-0.05)
            * max(We_L, 1e-30)**0.20
        )
        a_e = max(1e-8, min(a_t * (1.0 - math.exp(wet_exp)), a_t))

        Sc_L = mu_L / (rho_L * DL)
        k_L = (
            0.0051
            * (mu_L * g / rho_L)**(1.0 / 3.0)
            * (Lp / (a_e * mu_L))**(2.0 / 3.0)
            * Sc_L**(-0.5)
            * (a_t * dp)**0.4
        )

        Sc_G = self.mu_G / (self.rho_G * DG)
        k_G = (
            5.23
            * (a_t * DG / (R * self.T))
            * Re_G**0.7
            * Sc_G**(1.0 / 3.0)
            * (a_t * dp)**(-2.0)
        )

        H = henry_constant_Pa_m3_mol(self.T, component)
        K_G = 1.0 / (1.0 / k_G + H / k_L)
        m_x = H * (rho_L / WATER_MW) / self.P

        # Requested diagnostics
        G_flux_mol = self.gas_molar_flow() / self.A
        absorption_factor = self.liquid_molar_flow() / max(m_x * self.gas_molar_flow(), 1e-30)
        HTU_OG = G_flux_mol / max(K_G * a_e * self.P, 1e-30)
        NTU_OG = self.Z / max(HTU_OG, 1e-30)

        # Film-resistance fractions, useful for interpretation.
        total_resistance = 1.0 / K_G
        gas_resistance_fraction = (1.0 / k_G) / total_resistance
        liquid_resistance_fraction = (H / k_L) / total_resistance

        return {
            'a_t': a_t,
            'a_e': a_e,
            'wetting_fraction': a_e / a_t,
            'k_L': k_L,
            'k_G': k_G,
            'K_G': K_G,
            'H': H,
            'm_x': m_x,
            'Re_L': Re_L,
            'Re_G': Re_G,
            'Sc_L': Sc_L,
            'Sc_G': Sc_G,
            'L_mass_flux': Lp,
            'G_mass_flux': Gp,
            'absorption_factor': absorption_factor,
            'HTU_OG_m': HTU_OG,
            'NTU_OG': NTU_OG,
            'gas_resistance_fraction': gas_resistance_fraction,
            'liquid_resistance_fraction': liquid_resistance_fraction,
        }

    def solve_component(self, component, Z=None, return_profile=False):
        if Z is None:
            Z = self.Z

        G_flux = self.gas_molar_flow() / self.A
        L_flux = self.liquid_molar_flow() / self.A
        y_in = self.y_in_components[component]

        if y_in <= 0:
            result = {
                'component': component, 'y_in': 0.0, 'y_out': 0.0,
                'x_bottom': 0.0, 'removal': 0.0,
                'outlet_TOC_N': 0.0, 'coefficients': self.coefficients(component)
            }
            if return_profile:
                result.update({'z': np.array([0.0, Z]), 'y_profile': np.zeros(2), 'x_profile': np.zeros(2)})
            return result

        c = self.coefficients(component)
        a_e, K_G, m_x = c['a_e'], c['K_G'], c['m_x']
        rate_constant = K_G * a_e * self.P

        def integrate(x_bottom):
            def ode(z, state):
                y, x = state
                rate = rate_constant * (y - m_x * x)
                return [-rate / G_flux, -rate / L_flux]

            return solve_ivp(
                ode, [0.0, Z], [y_in, x_bottom],
                rtol=1e-8, atol=1e-12, max_step=max(Z / 100.0, 1e-5)
            )

        def boundary_error(x_bottom):
            sol_trial = integrate(x_bottom)
            return sol_trial.y[1, -1]

        x_eq = y_in / max(m_x, 1e-30)
        upper = max(1e-12, 2.0 * x_eq)
        try:
            x_bottom = brentq(boundary_error, 0.0, upper)
        except ValueError as exc:
            raise RuntimeError(
                f"{component}: counter-current boundary condition could not be solved. "
                f"x_bottom bracket [0, {upper:.3e}] may be insufficient."
            ) from exc

        sol = integrate(x_bottom)
        y_out = max(0.0, sol.y[0, -1])
        removal = 1.0 - y_out / y_in

        result = {
            'component': component,
            'y_in': y_in,
            'y_out': y_out,
            'x_bottom': x_bottom,
            'removal': removal,
            'outlet_TOC_N': self.y_to_component_concentration(y_out, component, 'mgVOC/Nm³'),
            'coefficients': c,
        }
        if return_profile:
            result.update({'z': sol.t, 'y_profile': sol.y[0], 'x_profile': sol.y[1]})
        return result

    # --------------------------------------------------------
    # HYDRAULICS
    # --------------------------------------------------------
    def packing_factor(self):
        p = PACKING_DATA[self.Dolgu_Tipi]
        if p.get('Fp_ft') is not None:
            return {
                'Fp_ft': float(p['Fp_ft']),
                'basis': p.get('Fp_basis', 'empirical'),
                'estimated': False,
            }
        Fp_m = p['a'] / max(p['epsilon'], 1e-12)**3
        return {
            'Fp_ft': Fp_m / FT_PER_M,
            'basis': 'geometric a/eps^3 fallback',
            'estimated': True,
        }

    @staticmethod
    def gpdc_flood_curve_cp(F_LV):
        if F_LV <= 0:
            raise ValueError("F_LV must be positive.")
        x = math.log10(F_LV)
        return 0.0394*x**3 + 0.0552*x**2 - 0.7634*x + 0.7863

    def pressure_drop_model(self):
        p = PACKING_DATA[self.Dolgu_Tipi]
        a, eps, psi = p['a'], p['epsilon'], p['psi']
        rhoL, muL, rhoG = self.water['rho'], self.water['mu'], self.rho_G
        Ug, Lp = self.gas_superficial_velocity(), self.liquid_mass_flux()
        ReL = Lp / (a * muL)

        hL = max(
            1e-6,
            min((12.0 * muL * (Lp / rhoL) * a**2 / (g * rhoL))**(1.0/3.0), 0.90*eps)
        )
        dPdry = psi * (a / max(eps, 1e-12)**3) * (rhoG * Ug**2 / 2.0)
        dPwet = dPdry * (eps / max(eps - hL, 1e-12))**3
        return {
            'ReL_hydraulic': ReL,
            'hL': hL,
            'dPdry_Pa_m': dPdry,
            'dPwet_Pa_m': dPwet,
            'dPwet_mbar_m': dPwet * MBAR_PER_PA,
            'Total_dP_Pa': dPwet * self.Z,
            'Total_dP_mbar': dPwet * self.Z * MBAR_PER_PA,
        }

    def flooding_model(self):
        rhoL, muL, rhoG = self.water['rho'], self.water['mu'], self.rho_G
        Ug_oper, Lp, Gp_oper = self.gas_superficial_velocity(), self.liquid_mass_flux(), self.gas_mass_flux()
        fp = self.packing_factor()
        Fp_ft = fp['Fp_ft']
        nu_cSt = (muL / rhoL) * 1e6
        K_flv = (Lp / rhoG) * math.sqrt(rhoG / rhoL)
        FLV_MIN, FLV_MAX = 0.01, 8.0
        U_lo = max(1e-4, K_flv / FLV_MAX)
        U_hi = min(50.0, max(U_lo * 1.01, K_flv / FLV_MIN))

        def residual(U_flood_m_s):
            Gf = rhoG * U_flood_m_s
            F_LV = (Lp / Gf) * math.sqrt(rhoG / rhoL)
            CP_corr = self.gpdc_flood_curve_cp(F_LV)
            U_flood_ft_s = U_flood_m_s * FT_PER_M
            C_s = U_flood_ft_s * math.sqrt(rhoG / max(rhoL - rhoG, 1e-30))
            CP_trial = C_s * math.sqrt(Fp_ft) * nu_cSt**0.05
            return CP_trial - CP_corr

        grid = np.geomspace(U_lo, U_hi, 400)
        valid_pairs = []
        prev_u, prev_f = grid[0], residual(grid[0])
        for u in grid[1:]:
            f = residual(u)
            if np.isfinite(prev_f) and np.isfinite(f) and prev_f*f <= 0:
                mid = math.sqrt(prev_u*u)
                F_mid = K_flv / mid
                if FLV_MIN <= F_mid <= FLV_MAX:
                    valid_pairs.append((prev_u, u))
            prev_u, prev_f = u, f

        if not valid_pairs:
            raise RuntimeError("No valid GPDC flood-velocity root found in the correlation range.")

        bracket = valid_pairs[-1]
        U_flood = brentq(residual, bracket[0], bracket[1], xtol=1e-10, rtol=1e-10)
        Gp_flood = rhoG * U_flood
        F_LV_flood = (Lp / Gp_flood) * math.sqrt(rhoG / rhoL)
        CP_flood = self.gpdc_flood_curve_cp(F_LV_flood)
        flooding_fraction = Ug_oper / max(U_flood, 1e-30)
        flooding_percent = 100.0 * flooding_fraction
        F_oper = Ug_oper * math.sqrt(rhoG)
        F_flood = U_flood * math.sqrt(rhoG)

        dP_flood_inH2O_ft = 0.115 * Fp_ft**0.7
        dP_flood_Pa_m = dP_flood_inH2O_ft * PA_PER_INH2O / M_PER_FT
        dP_flood_mbar_m = dP_flood_Pa_m * MBAR_PER_PA

        if flooding_percent < 40:
            regime = 'UNDERLOADED / HIGH CAPACITY MARGIN'
        elif flooding_percent < 60:
            regime = 'LOW-MODERATE HYDRAULIC LOADING'
        elif flooding_percent < 80:
            regime = 'NORMAL DESIGN REGION'
        elif flooding_percent < 90:
            regime = 'HIGH HYDRAULIC LOADING'
        elif flooding_percent < 100:
            regime = 'NEAR FLOODING'
        else:
            regime = 'FLOODING PREDICTED'

        return {
            'Ug': Ug_oper, 'U_flood': U_flood,
            'G_mass_flux': Gp_oper, 'G_flood_mass_flux': Gp_flood,
            'L_mass_flux': Lp, 'F_LV_flood': F_LV_flood, 'CP_flood': CP_flood,
            'Fp_ft': Fp_ft, 'Fp_basis': fp['basis'], 'Fp_estimated': fp['estimated'],
            'nu_L_cSt': nu_cSt, 'F_oper': F_oper, 'F_flood': F_flood,
            'flooding_fraction': flooding_fraction, 'flooding_percent': flooding_percent,
            'hydraulic_regime': regime,
            'dP_flood_inH2O_ft': dP_flood_inH2O_ft,
            'dP_flood_Pa_m': dP_flood_Pa_m,
            'dP_flood_mbar_m': dP_flood_mbar_m,
            'gpdc_valid': FLV_MIN <= F_LV_flood <= FLV_MAX,
        }

    def hydraulic_model(self):
        dp = self.pressure_drop_model()
        flood = self.flooding_model()
        dP_ratio = dp['dPwet_mbar_m'] / flood['dP_flood_mbar_m'] if flood['dP_flood_mbar_m'] > 0 else float('nan')
        return {**dp, **flood, 'dP_ratio_to_flood': dP_ratio}

    # --------------------------------------------------------
    # DIAGNOSTICS / SIMULATION
    # --------------------------------------------------------
    def diagnostics(self):
        warnings = []
        for comp in ('ACN', 'VAc'):
            c = self.coefficients(comp)
            if c['Re_L'] < 10:
                warnings.append(f"{comp}: Re_L={c['Re_L']:.2f} (low liquid loading / incomplete wetting risk)")
            if c['Re_G'] < 50:
                warnings.append(f"{comp}: Re_G={c['Re_G']:.2f} (gas-side correlation is in a low-Reynolds region)")
            if c['wetting_fraction'] < 0.30:
                warnings.append(f"{comp}: effective wetting={100*c['wetting_fraction']:.1f}% (poor distribution risk)")
            if c['absorption_factor'] <= 1.0:
                warnings.append(f"{comp}: absorption factor A={c['absorption_factor']:.3f} ≤ 1; absorption is thermodynamically/operationally difficult.")

        h = self.hydraulic_model()
        if h['Fp_estimated']:
            warnings.append("HYDRAULICS: no empirical GPDC packing factor; geometric fallback is being used.")
        if not h['gpdc_valid']:
            warnings.append(f"HYDRAULICS: F_LV={h['F_LV_flood']:.4f} is outside the intended GPDC range.")
        if h['flooding_percent'] >= 100:
            warnings.append("HYDRAULICS: GPDC predicts flooding.")
        elif h['flooding_percent'] >= 90:
            warnings.append("HYDRAULICS: column is very close to flooding (>90% flood).")
        elif h['flooding_percent'] >= 80:
            warnings.append("HYDRAULICS: high hydraulic loading (>80% flood).")

        liquid_loading = self.L / 3600.0 / self.A
        if liquid_loading < 1.0:
            warnings.append(
                f"Minimum wetting screening: liquid flux={liquid_loading:.3f} kg/m².s; verify packing-specific MWR."
            )
        if not (0.0 <= self.T_C <= 100.0):
            warnings.append("Water-property correlations are being extrapolated outside the normal liquid-water design range.")
        return warnings

    def simulate(self, return_profiles=False):
        ACN = self.solve_component('ACN', return_profile=return_profiles)
        VAc = self.solve_component('VAc', return_profile=return_profiles)
        y_out = {'ACN': ACN['y_out'], 'VAc': VAc['y_out']}
        outlet_by_basis = self.total_concentrations_from_y(y_out)

        inlet_selected = self.inlet_by_basis[self.concentration_basis]
        outlet_selected = outlet_by_basis[self.concentration_basis]
        overall_removal = 1.0 - outlet_selected / inlet_selected if inlet_selected > 0 else 0.0

        hydraulics = self.hydraulic_model()
        return {
            'concentration_basis': self.concentration_basis,
            'inlet_concentration': inlet_selected,
            'outlet_concentration': outlet_selected,
            'inlet_by_basis': dict(self.inlet_by_basis),
            'outlet_by_basis': outlet_by_basis,
            # backward-compatible mgVOC fields
            'inlet_TOC_N': self.inlet_by_basis['mgVOC/Nm³'],
            'outlet_TOC_N': outlet_by_basis['mgVOC/Nm³'],
            'overall_removal': overall_removal,
            'ACN_in': self.y_to_component_concentration(ACN['y_in'], 'ACN', self.concentration_basis),
            'ACN_out': self.y_to_component_concentration(ACN['y_out'], 'ACN', self.concentration_basis),
            'ACN_removal': ACN['removal'],
            'VAc_in': self.y_to_component_concentration(VAc['y_in'], 'VAc', self.concentration_basis),
            'VAc_out': self.y_to_component_concentration(VAc['y_out'], 'VAc', self.concentration_basis),
            'VAc_removal': VAc['removal'],
            'ACN': ACN, 'VAc': VAc,
            'water_properties': dict(self.water),
            **hydraulics,
            'warnings': self.diagnostics(),
        }

    def outlet_at_height(self, Z, basis=None):
        if basis is None:
            basis = self.concentration_basis
        ACN = self.solve_component('ACN', Z=Z)
        VAc = self.solve_component('VAc', Z=Z)
        y_out = {'ACN': ACN['y_out'], 'VAc': VAc['y_out']}
        return self.total_concentrations_from_y(y_out)[basis]

    def required_height(self, target_TOC=None, Zmax=20.0, basis=None):
        if basis is None:
            basis = self.concentration_basis
        if target_TOC is None:
            target_TOC = Hedef_Outlet
        inlet_value = self.inlet_by_basis[basis]
        if target_TOC >= inlet_value:
            return 0.0
        if self.outlet_at_height(Zmax, basis) > target_TOC:
            return None
        return brentq(lambda z: self.outlet_at_height(z, basis) - target_TOC, 1e-5, Zmax)


# Alias keeps old imports working while exposing the new engine.
ScrubberSimulationV3 = ScrubberSimulationV34


if __name__ == '__main__':
    sim = ScrubberSimulationV34()
    result = sim.simulate()
    z_req = sim.required_height(Hedef_Outlet)
    print(f"Outlet ({sim.concentration_basis}): {result['outlet_concentration']:.4f}")
    print(f"Removal: {100*result['overall_removal']:.3f}%")
    print(f"Required height: {z_req}")
    print(f"Flooding: {result['flooding_percent']:.3f}%")
    for comp in ('ACN', 'VAc'):
        c = result[comp]['coefficients']
        print(comp, 'A=', c['absorption_factor'], 'HTU=', c['HTU_OG_m'], 'NTU=', c['NTU_OG'])
