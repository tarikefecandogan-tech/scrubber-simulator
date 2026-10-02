"""
Scrubber Simulation V3 — engineering calculation engine.

This module contains the calculation model only. The Streamlit UI is in app.py.
Model basis: ACN/VAc absorption into water, ODE-based counter-current mass transfer,
separate pressure-drop model and GPDC-based flooding capacity estimate.
"""

# ============================================================
# 1. GİRDİ PARAMETRELERİ (ARAYÜZ KONSOLU)
# ============================================================

# @markdown ### 📌 Kolon ve Dolgu Parametreleri
Dolgu_Tipi = '25mm Metal Pall Ring' # @param ["25mm Metal Pall Ring", "38mm Metal Pall Ring", "IMTP #25 (Metal)", "IMTP #40 (Metal)", "CMR #2 (Metal)", "CMR #3 (Metal)", "13mm Super Raschig (MSRT-02)", "16mm Super Raschig (MSRT-03)", "25mm Super Raschig (MSRT-05)", "38mm Super Raschig (MSRT-06)"]
Kolon_Capi_D = 0.5 # @param {type:"number"}
Dolgu_Yuksekligi_Z = 1.4 # @param {type:"number"}

# @markdown ### ⚙️ Operasyon Koşulları
Sivi_Debisi_L = 2500 # @param {type:"number"}
Gaz_Debisi_Q = 117.53 # @param {type:"number"}
Debi_Referansi = 'Aktual' # @param ["Aktual", "Normal"]
Sicaklik_T = 22 # @param {type:"number"}
P_operating = 101325.0 # @param {type:"number"}

# @markdown ### 🧪 Besleme ve Tasarım Hedefi (Design Mode)
Giris_TOC_N = 5000 # @param {type:"number"}
ACN_Kutle_Kesri = 0.93 # @param {type:"number"}
VAc_Kutle_Kesri = 0.07 # @param {type:"number"}
Hedef_Outlet_TOC_N = 20 # @param {type:"number"}

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ============================================================
# 2. SABİT VERİTABANLARI
# ============================================================
T_N = 273.15
P_N = 101325.0
R = 8.314462618
g = 9.81
FT_PER_M = 3.280839895
PA_PER_INH2O = 249.08891
M_PER_FT = 0.3048
MBAR_PER_PA = 0.01

WATER = {
    'MW': 18.015e-3,
    'rho': 997.0,
    'mu': 0.00089,
    'sigma': 0.072
}

GAS = {
    'MW': 0.029,
    'mu_ref': 1.716e-5,
    'T_ref': 273.15,
    'S': 110.4
}

VOC = {
    'ACN': {
        'MW': 53.06e-3,
        'H25_atm_m3_mol': 1.18e-5,
        'DH_H_over_R': 4200.0,
        'DL': 1.0e-9,
        'DG': 1.0e-5
    },
    'VAc': {
        'MW': 86.09e-3,
        'H25_atm_m3_mol': 5.11e-4,
        'DH_H_over_R': 4500.0,
        'DL': 0.9e-9,
        'DG': 0.9e-5
    }
}

# Fp_ft: GPDC/Eckert için ampirik packing factor [ft^-1]
# Bilinen packings için kaynaklı ampirik değerler girildi.
# None olan packings için kod geometrik a/eps^3 yaklaşımına düşer ve UYARI verir.
PACKING_DATA = {
    '25mm Metal Pall Ring': {
        'a': 212.0, 'd': 0.025, 'epsilon': 0.962,
        'sigma_c': 0.075, 'psi': 1.20,
        'Fp_ft': 48.0,
        'Fp_basis': 'empirical'
    },
    '38mm Metal Pall Ring': {
        'a': 145.0, 'd': 0.038, 'epsilon': 0.967,
        'sigma_c': 0.075, 'psi': 1.00,
        'Fp_ft': 28.0,
        'Fp_basis': 'empirical'
    },
    'IMTP #25 (Metal)': {
        'a': 226.0, 'd': 0.025, 'epsilon': 0.970,
        'sigma_c': 0.075, 'psi': 1.10,
        'Fp_ft': 41.0,
        'Fp_basis': 'empirical'
    },
    'IMTP #40 (Metal)': {
        'a': 150.0, 'd': 0.040, 'epsilon': 0.975,
        'sigma_c': 0.075, 'psi': 0.90,
        'Fp_ft': 24.0,
        'Fp_basis': 'empirical'
    },
    'CMR #2 (Metal)': {
        'a': 188.0, 'd': 0.025, 'epsilon': 0.967,
        'sigma_c': 0.075, 'psi': 1.00,
        'Fp_ft': None,
        'Fp_basis': 'geometric_fallback'
    },
    'CMR #3 (Metal)': {
        'a': 135.0, 'd': 0.040, 'epsilon': 0.972,
        'sigma_c': 0.075, 'psi': 0.80,
        'Fp_ft': None,
        'Fp_basis': 'geometric_fallback'
    },
    '13mm Super Raschig (MSRT-02)': {
        'a': 250.0, 'd': 0.013, 'epsilon': 0.965,
        'sigma_c': 0.075, 'psi': 1.0,
        'Fp_ft': None,
        'Fp_basis': 'geometric_fallback'
    },
    '16mm Super Raschig (MSRT-03)': {
        'a': 215.0, 'd': 0.016, 'epsilon': 0.961,
        'sigma_c': 0.075, 'psi': 0.9,
        'Fp_ft': None,
        'Fp_basis': 'geometric_fallback'
    },
    '25mm Super Raschig (MSRT-05)': {
        'a': 150.0, 'd': 0.025, 'epsilon': 0.972,
        'sigma_c': 0.075, 'psi': 0.8,
        'Fp_ft': None,
        'Fp_basis': 'geometric_fallback'
    },
    '38mm Super Raschig (MSRT-06)': {
        'a': 120.0, 'd': 0.038, 'epsilon': 0.978,
        'sigma_c': 0.075, 'psi': 0.7,
        'Fp_ft': None,
        'Fp_basis': 'geometric_fallback'
    }
}


def hava_viskozitesi(T_kelvin):
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
    H_atm = H25 * math.exp(
        DH_over_R * (1.0 / 298.15 - 1.0 / T_kelvin)
    )
    return H_atm * P_N


class ScrubberSimulationV3:
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
        Giris_TOC_N=Giris_TOC_N,
        ACN_Kutle_Kesri=ACN_Kutle_Kesri,
        VAc_Kutle_Kesri=VAc_Kutle_Kesri
    ):
        self.Dolgu_Tipi = Dolgu_Tipi
        self.D = Kolon_Capi_D
        self.Z = Dolgu_Yuksekligi_Z
        self.L = Sivi_Debisi_L
        self.Q_input = Gaz_Debisi_Q
        self.Debi_Referansi = Debi_Referansi
        self.T_C = Sicaklik_T
        self.T = Sicaklik_T + 273.15
        self.P = P_operating
        self.TOC_in = Giris_TOC_N
        self.w_ACN = ACN_Kutle_Kesri
        self.w_VAc = VAc_Kutle_Kesri

        if self.Dolgu_Tipi not in PACKING_DATA:
            raise ValueError(f"Bilinmeyen dolgu tipi: {self.Dolgu_Tipi}")

        if abs(self.w_ACN + self.w_VAc - 1.0) > 1e-8:
            raise ValueError("ACN + VAc kütle kesirleri toplamı 1.0 olmalıdır.")

        if self.D <= 0 or self.Z <= 0 or self.L <= 0 or self.Q_input <= 0:
            raise ValueError("Tüm fiziksel girdiler sıfırdan büyük olmalıdır.")

        if self.P <= 0:
            raise ValueError("İşletme basıncı sıfırdan büyük olmalıdır.")

        self.A = math.pi * self.D**2 / 4.0
        self.rho_G = gas_density(self.P, self.T)
        self.mu_G = hava_viskozitesi(self.T)

        if self.Debi_Referansi.lower().startswith('normal'):
            self.Q = self.Q_input * (self.T / T_N) * (P_N / self.P)
        else:
            self.Q = self.Q_input

    # --------------------------------------------------------
    # TEMEL AKIŞ FONKSİYONLARI
    # --------------------------------------------------------
    def gas_molar_flow(self):
        return self.Q * self.P / (R * self.T) / 3600.0

    def liquid_molar_flow(self):
        return self.L / WATER['MW'] / 3600.0

    def gas_mass_flux(self):
        return self.Q * self.rho_G / 3600.0 / self.A

    def liquid_mass_flux(self):
        return self.L / 3600.0 / self.A

    def gas_superficial_velocity(self):
        return self.Q / 3600.0 / self.A

    # --------------------------------------------------------
    # TOC <-> MOL KESRİ
    # --------------------------------------------------------
    def toc_to_y(self, TOC_N, component, mass_fraction):
        MW = VOC[component]['MW']
        C_mass_kg_Nm3 = TOC_N * mass_fraction * 1e-6
        C_molar_Nm3 = C_mass_kg_Nm3 / MW
        Cgas_N = P_N / (R * T_N)
        return C_molar_Nm3 / Cgas_N

    def y_to_toc(self, y, component):
        MW = VOC[component]['MW']
        Cgas_N = P_N / (R * T_N)
        return y * Cgas_N * MW * 1e6

    # --------------------------------------------------------
    # KÜTLE TRANSFER KATSAYILARI
    # --------------------------------------------------------
    def coefficients(self, component):
        p = PACKING_DATA[self.Dolgu_Tipi]
        a_t = p['a']
        dp = p['d']
        sigma_c = p['sigma_c']

        rho_L = WATER['rho']
        mu_L = WATER['mu']
        sigma_L = WATER['sigma']
        DL = VOC[component]['DL']
        DG = VOC[component]['DG']

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

        a_e = max(
            1e-8,
            min(a_t * (1.0 - math.exp(wet_exp)), a_t)
        )

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
        m_x = H * (WATER['rho'] / WATER['MW']) / self.P

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
            'G_mass_flux': Gp
        }

    # --------------------------------------------------------
    # COUNTER-CURRENT ODE ÇÖZÜMÜ
    # --------------------------------------------------------
    def solve_component(self, component, Z=None, return_profile=False):
        if Z is None:
            Z = self.Z

        G_flux = self.gas_molar_flow() / self.A
        L_flux = self.liquid_molar_flow() / self.A
        wf = self.w_ACN if component == 'ACN' else self.w_VAc
        y_in = self.toc_to_y(self.TOC_in, component, wf)

        c = self.coefficients(component)
        a_e = c['a_e']
        K_G = c['K_G']
        m_x = c['m_x']
        rate_constant = K_G * a_e * self.P

        def integrate(x_bottom):
            def ode(z, state):
                y, x = state
                rate = rate_constant * (y - m_x * x)
                return [-rate / G_flux, -rate / L_flux]

            return solve_ivp(
                ode,
                [0.0, Z],
                [y_in, x_bottom],
                rtol=1e-8,
                atol=1e-12,
                max_step=max(Z / 100.0, 1e-5)
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
                f"{component}: Counter-current boundary condition çözülemedi. "
                f"x_bottom bracket'i [0, {upper:.3e}] yetersiz olabilir."
            ) from exc

        sol = integrate(x_bottom)
        y_out = max(0.0, sol.y[0, -1])
        removal = (1.0 - y_out / y_in) if y_in > 0 else 0.0

        result = {
            'component': component,
            'y_in': y_in,
            'y_out': y_out,
            'x_bottom': x_bottom,
            'removal': removal,
            'outlet_TOC_N': self.y_to_toc(y_out, component),
            'coefficients': c
        }

        if return_profile:
            result['z'] = sol.t
            result['y_profile'] = sol.y[0]
            result['x_profile'] = sol.y[1]

        return result

    # ========================================================
    # 3. YENİ HİDROLİK MODEL
    # ========================================================

    def packing_factor(self):
        """
        GPDC/Eckert ampirik packing factor [ft^-1].
        Veri tabanında ampirik değer yoksa a/eps^3 geometrik yaklaşımı
        yalnızca fallback olarak kullanılır.
        """
        p = PACKING_DATA[self.Dolgu_Tipi]

        if p.get('Fp_ft') is not None:
            return {
                'Fp_ft': float(p['Fp_ft']),
                'basis': p.get('Fp_basis', 'empirical'),
                'estimated': False
            }

        # a/eps^3 [m^-1] -> [ft^-1]
        Fp_m = p['a'] / max(p['epsilon'], 1e-12)**3
        Fp_ft = Fp_m / FT_PER_M

        return {
            'Fp_ft': Fp_ft,
            'basis': 'geometric a/eps^3 fallback',
            'estimated': True
        }

    @staticmethod
    def gpdc_flood_curve_cp(F_LV):
        """
        Random packing için GPDC flooding-line curve fit.
        CP = f(log10(F_LV))

        Yaklaşık geçerlilik alanı: 0.01 <= F_LV <= 8.
        """
        if F_LV <= 0:
            raise ValueError("F_LV sıfırdan büyük olmalıdır.")

        x = math.log10(F_LV)
        return (
            0.0394 * x**3
            + 0.0552 * x**2
            - 0.7634 * x
            + 0.7863
        )

    def pressure_drop_model(self):
        """
        Mevcut modelin wet/dry pressure-drop yaklaşımını korur.
        Flooding yüzdesi BU FONKSİYONDAN türetilmez.
        """
        p = PACKING_DATA[self.Dolgu_Tipi]
        a = p['a']
        eps = p['epsilon']
        psi = p['psi']

        rhoL = WATER['rho']
        muL = WATER['mu']
        rhoG = self.rho_G
        Ug = self.gas_superficial_velocity()
        Lp = self.liquid_mass_flux()

        ReL = Lp / (a * muL)

        # Basitleştirilmiş sıvı tutunması yaklaşımı
        hL = max(
            1e-6,
            min(
                (12.0 * muL * (Lp / rhoL) * a**2 / (g * rhoL))**(1.0 / 3.0),
                0.90 * eps
            )
        )

        # Dry / wet specific pressure drop [Pa/m]
        dPdry = psi * (a / max(eps, 1e-12)**3) * (rhoG * Ug**2 / 2.0)
        dPwet = dPdry * (eps / max(eps - hL, 1e-12))**3

        return {
            'ReL_hydraulic': ReL,
            'hL': hL,
            'dPdry_Pa_m': dPdry,
            'dPwet_Pa_m': dPwet,
            'dPwet_mbar_m': dPwet * MBAR_PER_PA,
            'Total_dP_Pa': dPwet * self.Z,
            'Total_dP_mbar': dPwet * self.Z * MBAR_PER_PA
        }

    def flooding_model(self):
        """
        GPDC/Eckert tabanlı bağımsız flooding kapasitesi.

        Ana fikir:
          1) Trial U_flood seç
          2) G_flood = rho_G * U_flood
          3) F_LV = (L/G_flood)*sqrt(rho_G/rho_L)
          4) GPDC'den CP_corr(F_LV)
          5) CP_trial = Cs*sqrt(Fp)*nu^0.05
          6) CP_trial - CP_corr = 0 kökü -> U_flood
          7) %Flood = U_oper/U_flood*100

        Kister-Gill dP_flood ayrıca diagnostik olarak hesaplanır;
        %Flood hesabında kullanılmaz.
        """
        rhoL = WATER['rho']
        muL = WATER['mu']
        rhoG = self.rho_G
        Ug_oper = self.gas_superficial_velocity()
        Lp = self.liquid_mass_flux()
        Gp_oper = self.gas_mass_flux()

        fp = self.packing_factor()
        Fp_ft = fp['Fp_ft']

        # Kinematik viskozite [cSt]
        nu_cSt = (muL / rhoL) * 1e6

        # GPDC rating denkleminde F_LV, flood gaz akısına bağlıdır.
        # F_LV = K / U_flood
        K_flv = (Lp / rhoG) * math.sqrt(rhoG / rhoL)

        # GPDC curve-fit geçerlilik alanından gelen U bracket'i
        FLV_MIN = 0.01
        FLV_MAX = 8.0
        U_min_valid = K_flv / FLV_MAX
        U_max_valid = K_flv / FLV_MIN

        # Fiziksel/nümerik güvenlik sınırları
        U_lo = max(1e-4, U_min_valid)
        U_hi = min(50.0, max(U_lo * 1.01, U_max_valid))

        def residual(U_flood_m_s):
            Gf = rhoG * U_flood_m_s
            F_LV = (Lp / Gf) * math.sqrt(rhoG / rhoL)
            CP_corr = self.gpdc_flood_curve_cp(F_LV)

            U_flood_ft_s = U_flood_m_s * FT_PER_M
            C_s = U_flood_ft_s * math.sqrt(rhoG / max(rhoL - rhoG, 1e-30))
            CP_trial = C_s * math.sqrt(Fp_ft) * nu_cSt**0.05

            return CP_trial - CP_corr

        # Tek bir kör bracket yerine validity alanında tarama yapıyoruz.
        grid = np.geomspace(U_lo, U_hi, 400)
        valid_pairs = []

        prev_u = grid[0]
        prev_f = residual(prev_u)

        for u in grid[1:]:
            f = residual(u)
            if np.isfinite(prev_f) and np.isfinite(f) and prev_f * f <= 0:
                # Kök çevresindeki F_LV'nin validity alanında olduğunu kontrol et
                mid = math.sqrt(prev_u * u)
                F_mid = K_flv / mid
                if FLV_MIN <= F_mid <= FLV_MAX:
                    valid_pairs.append((prev_u, u))
            prev_u, prev_f = u, f

        if not valid_pairs:
            raise RuntimeError(
                "GPDC flooding hızı için geçerli kök bulunamadı. "
                "Flow parameter GPDC geçerlilik alanının dışında olabilir."
            )

        # Fiziksel flood kökü için en yüksek hızlı geçerli kökü seç.
        # Bu, düşük hızlı matematiksel/spurious kökleri eler.
        bracket = valid_pairs[-1]
        U_flood = brentq(residual, bracket[0], bracket[1], xtol=1e-10, rtol=1e-10)

        Gp_flood = rhoG * U_flood
        F_LV_flood = (Lp / Gp_flood) * math.sqrt(rhoG / rhoL)
        CP_flood = self.gpdc_flood_curve_cp(F_LV_flood)

        flooding_fraction = Ug_oper / max(U_flood, 1e-30)
        flooding_percent = 100.0 * flooding_fraction

        F_oper = Ug_oper * math.sqrt(rhoG)
        F_flood = U_flood * math.sqrt(rhoG)

        # Kister-Gill flooding pressure-drop diagnostik değeri
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
            'Ug': Ug_oper,
            'U_flood': U_flood,
            'G_mass_flux': Gp_oper,
            'G_flood_mass_flux': Gp_flood,
            'L_mass_flux': Lp,
            'F_LV_flood': F_LV_flood,
            'CP_flood': CP_flood,
            'Fp_ft': Fp_ft,
            'Fp_basis': fp['basis'],
            'Fp_estimated': fp['estimated'],
            'nu_L_cSt': nu_cSt,
            'F_oper': F_oper,
            'F_flood': F_flood,
            'flooding_fraction': flooding_fraction,
            'flooding_percent': flooding_percent,
            'hydraulic_regime': regime,
            'dP_flood_inH2O_ft': dP_flood_inH2O_ft,
            'dP_flood_Pa_m': dP_flood_Pa_m,
            'dP_flood_mbar_m': dP_flood_mbar_m,
            'gpdc_valid': FLV_MIN <= F_LV_flood <= FLV_MAX
        }

    def hydraulic_model(self):
        dp = self.pressure_drop_model()
        flood = self.flooding_model()

        dP_ratio = (
            dp['dPwet_mbar_m'] / flood['dP_flood_mbar_m']
            if flood['dP_flood_mbar_m'] > 0
            else float('nan')
        )

        return {
            **dp,
            **flood,
            'dP_ratio_to_flood': dP_ratio
        }

    # --------------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------------
    def diagnostics(self):
        warnings = []

        for comp in ['ACN', 'VAc']:
            c = self.coefficients(comp)

            if c['Re_L'] < 10:
                warnings.append(
                    f"{comp}: Re_L={c['Re_L']:.2f} "
                    "(Çok düşük sıvı yükü, dolgular yeterince ıslanmayabilir)"
                )

            if c['Re_G'] < 50:
                warnings.append(
                    f"{comp}: Re_G={c['Re_G']:.2f} "
                    "(Gaz tarafı korelasyonu düşük Reynolds bölgesinde)"
                )

            if c['wetting_fraction'] < 0.30:
                warnings.append(
                    f"{comp}: Efektif Islanma = %{100*c['wetting_fraction']:.1f} "
                    "(Zayıf sıvı dağılımı riski)"
                )

        h = self.hydraulic_model()

        if h['Fp_estimated']:
            warnings.append(
                "HİDROLİK: Seçilen packing için ampirik GPDC packing factor yok; "
                "a/eps^3 geometrik fallback kullanıldı. Flooding sonucu yaklaşık kabul edilmelidir."
            )

        if not h['gpdc_valid']:
            warnings.append(
                f"HİDROLİK: F_LV={h['F_LV_flood']:.4f} GPDC önerilen validity alanının dışında."
            )

        if h['flooding_percent'] >= 100:
            warnings.append(
                "HİDROLİK: GPDC modeline göre kolon flooding kapasitesini aşıyor!"
            )
        elif h['flooding_percent'] >= 90:
            warnings.append(
                "HİDROLİK: Kolon flooding sınırına çok yakın (> %90 flood)."
            )
        elif h['flooding_percent'] >= 80:
            warnings.append(
                "HİDROLİK: Yüksek hidrolik yük (> %80 flood); tasarım marjı kontrol edilmeli."
            )

        liquid_loading = self.L / 3600.0 / self.A
        if liquid_loading < 1.0:
            warnings.append(
                f"Minimum Islatma: Sıvı akısı = {liquid_loading:.3f} kg/m2.s "
                "(Packing malzemesi ve minimum wetting rate ayrıca doğrulanmalı)"
            )

        return warnings

    # --------------------------------------------------------
    # ANA SİMÜLASYON
    # --------------------------------------------------------
    def simulate(self, return_profiles=False):
        ACN = self.solve_component('ACN', return_profile=return_profiles)
        VAc = self.solve_component('VAc', return_profile=return_profiles)

        outlet = ACN['outlet_TOC_N'] + VAc['outlet_TOC_N']
        removal = 1.0 - outlet / self.TOC_in
        hydraulics = self.hydraulic_model()

        return {
            'inlet_TOC_N': self.TOC_in,
            'outlet_TOC_N': outlet,
            'overall_removal': removal,
            'ACN_in': self.TOC_in * self.w_ACN,
            'ACN_out': ACN['outlet_TOC_N'],
            'ACN_removal': ACN['removal'],
            'VAc_in': self.TOC_in * self.w_VAc,
            'VAc_out': VAc['outlet_TOC_N'],
            'VAc_removal': VAc['removal'],
            'ACN': ACN,
            'VAc': VAc,
            **hydraulics,
            'warnings': self.diagnostics()
        }

    def required_height(self, target_TOC=None, Zmax=20.0):
        if target_TOC is None:
            target_TOC = Hedef_Outlet_TOC_N

        if target_TOC >= self.TOC_in:
            return 0.0

        def outlet_at_Z(Z):
            ACN = self.solve_component('ACN', Z=Z)
            VAc = self.solve_component('VAc', Z=Z)
            return ACN['outlet_TOC_N'] + VAc['outlet_TOC_N']

        if outlet_at_Z(Zmax) > target_TOC:
            return None

        return brentq(
            lambda z: outlet_at_Z(z) - target_TOC,
            1e-5,
            Zmax
        )
