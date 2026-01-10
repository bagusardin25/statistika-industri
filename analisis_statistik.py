"""
Analisis Statistik Data Penelitian Per Provinsi 2023
=====================================================
Script ini melakukan:
1. Uji-T (T-Test)
2. Uji-F (F-Test)
3. ANOVA
4. Regresi Linear dengan Visualisasi
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, ttest_1samp, levene, shapiro
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
warnings.filterwarnings('ignore')

# Set style untuk visualisasi
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Tingkat signifikansi
ALPHA = 0.05

# =====================================================
# DATA PENELITIAN
# =====================================================
data = {
    'No': list(range(1, 35)),
    'Provinsi': [
        'Aceh', 'Sumatera Utara', 'Sumatera Barat', 'Riau', 'Jambi',
        'Sumatera Selatan', 'Bengkulu', 'Lampung', 'Kep. Bangka Belitung',
        'Kepulauan Riau', 'DKI Jakarta', 'Jawa Barat', 'Jawa Tengah',
        'DI Yogyakarta', 'Jawa Timur', 'Banten', 'Bali', 'Nusa Tenggara Barat',
        'Nusa Tenggara Timur', 'Kalimantan Barat', 'Kalimantan Tengah',
        'Kalimantan Selatan', 'Kalimantan Timur', 'Kalimantan Utara',
        'Sulawesi Utara', 'Sulawesi Tengah', 'Sulawesi Selatan',
        'Sulawesi Tenggara', 'Gorontalo', 'Sulawesi Barat', 'Maluku',
        'Maluku Utara', 'Papua', 'Papua Barat'
    ],
    'Angka_Stunting': [
        29.4, 18.9, 23.6, 13.6, 13.5, 20.3, 20.2, 14.9, 20.6, 16.8,
        17.6, 21.7, 20.7, 18.0, 17.7, 24.0, 7.2, 24.6, 37.9, 24.5,
        23.5, 24.7, 22.9, 17.4, 21.3, 27.2, 27.4, 30.0, 26.9, 30.3,
        28.4, 23.7, 28.6, 24.8
    ],
    'Pertumbuhan_Ekonomi': [
        4.23, 5.01, 4.62, 4.21, 4.67, 5.08, 4.28, 4.55, 4.38, 5.16,
        4.96, 5.00, 4.97, 5.07, 4.95, 4.81, 5.71, 1.80, 3.47, 4.46,
        4.14, 4.84, 6.22, 4.94, 5.48, 11.91, 4.51, 5.35, 4.50, 5.23,
        5.21, 20.49, 5.18, 4.22
    ],
    'Tingkat_Pendidikan': [
        9.55, 9.82, 9.28, 9.32, 8.81, 8.50, 9.03, 8.29, 8.25, 10.41,
        11.45, 8.83, 8.01, 9.83, 8.11, 9.15, 9.45, 7.74, 7.82, 7.71,
        8.73, 8.55, 9.99, 9.34, 9.77, 8.96, 8.76, 9.31, 8.10, 8.13,
        10.20, 9.26, 7.15, 7.93
    ],
    'Akses_Sanitasi': [
        78.85, 84.18, 70.97, 84.58, 83.04, 80.54, 80.28, 84.58, 93.21,
        91.10, 93.50, 74.88, 85.20, 96.42, 83.72, 86.41, 95.70, 85.11,
        75.67, 79.89, 76.31, 82.89, 91.21, 84.22, 85.91, 75.80, 93.69,
        88.99, 81.72, 80.73, 78.17, 80.64, 43.00, 76.30
    ]
}

df = pd.DataFrame(data)

print("=" * 70)
print("ANALISIS STATISTIK DATA PENELITIAN PER PROVINSI 2023")
print("=" * 70)

# =====================================================
# 1. STATISTIK DESKRIPTIF
# =====================================================
print("\n" + "=" * 70)
print("1. STATISTIK DESKRIPTIF")
print("=" * 70)

stats_desc = df[['Angka_Stunting', 'Pertumbuhan_Ekonomi', 
                  'Tingkat_Pendidikan', 'Akses_Sanitasi']].describe()
print(stats_desc.round(3))

# =====================================================
# 2. UJI-T (T-TEST)
# =====================================================
print("\n" + "=" * 70)
print("2. UJI-T (T-TEST)")
print("=" * 70)

# 2.1 One-Sample T-Test
# Menguji apakah rata-rata stunting berbeda signifikan dari nilai tertentu (misal: 20%)
print("\n2.1 One-Sample T-Test")
print("-" * 50)
print("H0: Rata-rata angka stunting = 20%")
print("H1: Rata-rata angka stunting != 20%")

t_stat, p_value = ttest_1samp(df['Angka_Stunting'], 20)
print(f"\nStatistik T     : {t_stat:.4f}")
print(f"P-value         : {p_value:.4f}")
print(f"Keputusan (alpha={ALPHA}): {'Tolak H0' if p_value < ALPHA else 'Gagal Tolak H0'}")

# 2.2 Independent Two-Sample T-Test
# Membagi data menjadi 2 kelompok berdasarkan median pertumbuhan ekonomi
print("\n2.2 Independent Two-Sample T-Test")
print("-" * 50)
median_ekonomi = df['Pertumbuhan_Ekonomi'].median()
grup_rendah = df[df['Pertumbuhan_Ekonomi'] <= median_ekonomi]['Angka_Stunting']
grup_tinggi = df[df['Pertumbuhan_Ekonomi'] > median_ekonomi]['Angka_Stunting']

print(f"Median Pertumbuhan Ekonomi: {median_ekonomi:.2f}%")
print(f"Grup Rendah (n={len(grup_rendah)}): Mean Stunting = {grup_rendah.mean():.2f}%")
print(f"Grup Tinggi (n={len(grup_tinggi)}): Mean Stunting = {grup_tinggi.mean():.2f}%")

print("\nH0: Tidak ada perbedaan rata-rata stunting antara kedua grup")
print("H1: Ada perbedaan rata-rata stunting antara kedua grup")

t_stat2, p_value2 = ttest_ind(grup_rendah, grup_tinggi)
print(f"\nStatistik T     : {t_stat2:.4f}")
print(f"P-value         : {p_value2:.4f}")
print(f"Keputusan (alpha={ALPHA}): {'Tolak H0' if p_value2 < ALPHA else 'Gagal Tolak H0'}")

# =====================================================
# 3. UJI-F (F-TEST / LEVENE'S TEST)
# =====================================================
print("\n" + "=" * 70)
print("3. UJI-F (LEVENE'S TEST - UJI HOMOGENITAS VARIANS)")
print("=" * 70)

# Uji homogenitas varians antara grup
print("\nH0: Varians kedua grup sama (homogen)")
print("H1: Varians kedua grup berbeda (tidak homogen)")

f_stat, f_pvalue = levene(grup_rendah, grup_tinggi)
print(f"\nStatistik Levene: {f_stat:.4f}")
print(f"P-value         : {f_pvalue:.4f}")
print(f"Keputusan (alpha={ALPHA}): {'Tolak H0 - Varians tidak homogen' if f_pvalue < ALPHA else 'Gagal Tolak H0 - Varians homogen'}")

# Uji F tradisional (rasio varians)
print("\n" + "-" * 50)
print("Uji F Tradisional (Rasio Varians)")
print("-" * 50)
var1 = grup_rendah.var()
var2 = grup_tinggi.var()
f_ratio = var1 / var2 if var1 > var2 else var2 / var1
df1 = len(grup_rendah) - 1
df2 = len(grup_tinggi) - 1

print(f"Varians Grup Rendah : {var1:.4f}")
print(f"Varians Grup Tinggi : {var2:.4f}")
print(f"F-ratio             : {f_ratio:.4f}")
print(f"df1, df2            : {df1}, {df2}")

# =====================================================
# 4. ANOVA (ANALYSIS OF VARIANCE)
# =====================================================
print("\n" + "=" * 70)
print("4. ANOVA (ANALYSIS OF VARIANCE)")
print("=" * 70)

# Membagi provinsi menjadi 3 kategori berdasarkan tingkat stunting
df['Kategori_Stunting'] = pd.cut(df['Angka_Stunting'], 
                                  bins=[0, 20, 25, 100], 
                                  labels=['Rendah', 'Sedang', 'Tinggi'])

print("\nKategori Stunting:")
print(f"  - Rendah : <= 20% (n={len(df[df['Kategori_Stunting']=='Rendah'])})")
print(f"  - Sedang : 20-25% (n={len(df[df['Kategori_Stunting']=='Sedang'])})")
print(f"  - Tinggi : > 25%  (n={len(df[df['Kategori_Stunting']=='Tinggi'])})")

# 4.1 One-Way ANOVA untuk Pertumbuhan Ekonomi
print("\n4.1 One-Way ANOVA: Pertumbuhan Ekonomi berdasarkan Kategori Stunting")
print("-" * 50)
print("H0: Tidak ada perbedaan rata-rata pertumbuhan ekonomi antar kategori")
print("H1: Ada perbedaan rata-rata pertumbuhan ekonomi antar kategori")

grup_rendah_pe = df[df['Kategori_Stunting'] == 'Rendah']['Pertumbuhan_Ekonomi']
grup_sedang_pe = df[df['Kategori_Stunting'] == 'Sedang']['Pertumbuhan_Ekonomi']
grup_tinggi_pe = df[df['Kategori_Stunting'] == 'Tinggi']['Pertumbuhan_Ekonomi']

f_stat_anova, p_anova = f_oneway(grup_rendah_pe, grup_sedang_pe, grup_tinggi_pe)
print(f"\nStatistik F     : {f_stat_anova:.4f}")
print(f"P-value         : {p_anova:.4f}")
print(f"Keputusan (alpha={ALPHA}): {'Tolak H0' if p_anova < ALPHA else 'Gagal Tolak H0'}")

# 4.2 One-Way ANOVA untuk Tingkat Pendidikan
print("\n4.2 One-Way ANOVA: Tingkat Pendidikan berdasarkan Kategori Stunting")
print("-" * 50)

grup_rendah_tp = df[df['Kategori_Stunting'] == 'Rendah']['Tingkat_Pendidikan']
grup_sedang_tp = df[df['Kategori_Stunting'] == 'Sedang']['Tingkat_Pendidikan']
grup_tinggi_tp = df[df['Kategori_Stunting'] == 'Tinggi']['Tingkat_Pendidikan']

f_stat_tp, p_tp = f_oneway(grup_rendah_tp, grup_sedang_tp, grup_tinggi_tp)
print(f"Statistik F     : {f_stat_tp:.4f}")
print(f"P-value         : {p_tp:.4f}")
print(f"Keputusan (alpha={ALPHA}): {'Tolak H0' if p_tp < ALPHA else 'Gagal Tolak H0'}")

# 4.3 One-Way ANOVA untuk Akses Sanitasi
print("\n4.3 One-Way ANOVA: Akses Sanitasi berdasarkan Kategori Stunting")
print("-" * 50)

grup_rendah_as = df[df['Kategori_Stunting'] == 'Rendah']['Akses_Sanitasi']
grup_sedang_as = df[df['Kategori_Stunting'] == 'Sedang']['Akses_Sanitasi']
grup_tinggi_as = df[df['Kategori_Stunting'] == 'Tinggi']['Akses_Sanitasi']

f_stat_as, p_as = f_oneway(grup_rendah_as, grup_sedang_as, grup_tinggi_as)
print(f"Statistik F     : {f_stat_as:.4f}")
print(f"P-value         : {p_as:.4f}")
print(f"Keputusan (alpha={ALPHA}): {'Tolak H0' if p_as < ALPHA else 'Gagal Tolak H0'}")

# ANOVA Table menggunakan statsmodels
print("\n4.4 Tabel ANOVA Lengkap (dengan statsmodels)")
print("-" * 50)
model = ols('Angka_Stunting ~ C(Kategori_Stunting)', data=df).fit()
anova_table = anova_lm(model, typ=2)
print(anova_table.round(4))

# =====================================================
# 5. REGRESI LINEAR
# =====================================================
print("\n" + "=" * 70)
print("5. ANALISIS REGRESI LINEAR")
print("=" * 70)

# 5.1 Regresi Sederhana: Stunting vs Pertumbuhan Ekonomi
print("\n5.1 Regresi Linear Sederhana: Stunting ~ Pertumbuhan Ekonomi")
print("-" * 50)

X1 = sm.add_constant(df['Pertumbuhan_Ekonomi'])
model1 = sm.OLS(df['Angka_Stunting'], X1).fit()
print(model1.summary())

# 5.2 Regresi Sederhana: Stunting vs Tingkat Pendidikan
print("\n5.2 Regresi Linear Sederhana: Stunting ~ Tingkat Pendidikan")
print("-" * 50)

X2 = sm.add_constant(df['Tingkat_Pendidikan'])
model2 = sm.OLS(df['Angka_Stunting'], X2).fit()
print(model2.summary())

# 5.3 Regresi Sederhana: Stunting vs Akses Sanitasi
print("\n5.3 Regresi Linear Sederhana: Stunting ~ Akses Sanitasi")
print("-" * 50)

X3 = sm.add_constant(df['Akses_Sanitasi'])
model3 = sm.OLS(df['Angka_Stunting'], X3).fit()
print(model3.summary())

# 5.4 Regresi Berganda
print("\n5.4 Regresi Linear Berganda: Stunting ~ Semua Variabel")
print("-" * 50)

X_multi = df[['Pertumbuhan_Ekonomi', 'Tingkat_Pendidikan', 'Akses_Sanitasi']]
X_multi = sm.add_constant(X_multi)
model_multi = sm.OLS(df['Angka_Stunting'], X_multi).fit()
print(model_multi.summary())

# =====================================================
# 5.5 UJI SIMULTAN (UJI F) DAN UJI PARSIAL (UJI T)
# =====================================================
print("\n" + "=" * 70)
print("5.5 UJI SIMULTAN (UJI F) DAN UJI PARSIAL (UJI T)")
print("=" * 70)

# 5.5.1 Uji Simultan (Uji F)
print("\n5.5.1 UJI SIMULTAN (UJI F)")
print("-" * 50)
print("H0: b1 = b2 = b3 = 0 (Semua variabel independen tidak berpengaruh)")
print("H1: Minimal satu bi != 0 (Minimal satu variabel berpengaruh)")

f_statistic = model_multi.fvalue
f_pvalue = model_multi.f_pvalue
df_model = model_multi.df_model  # k (jumlah variabel independen)
df_resid = model_multi.df_resid  # n - k - 1

print(f"\nF-hitung        : {f_statistic:.4f}")
print(f"F-tabel (alpha={ALPHA}, df1={int(df_model)}, df2={int(df_resid)}): {stats.f.ppf(1-ALPHA, df_model, df_resid):.4f}")
print(f"P-value         : {f_pvalue:.6f}")
print(f"\nKeputusan (α={ALPHA}):")
if f_pvalue < ALPHA:
    print(f"   F-hitung ({f_statistic:.4f}) > F-tabel ({stats.f.ppf(1-ALPHA, df_model, df_resid):.4f})")
    print(f"   P-value ({f_pvalue:.6f}) < α ({ALPHA})")
    print("   >>> TOLAK H0: Model regresi signifikan secara simultan <<<")
    print("   >>> Variabel independen secara bersama-sama berpengaruh terhadap Stunting <<<")
else:
    print(f"   F-hitung ({f_statistic:.4f}) <= F-tabel ({stats.f.ppf(1-ALPHA, df_model, df_resid):.4f})")
    print(f"   P-value ({f_pvalue:.6f}) >= α ({ALPHA})")
    print("   >>> GAGAL TOLAK H0: Model regresi tidak signifikan <<<")

# 5.5.2 Uji Parsial (Uji t)
print("\n5.5.2 UJI PARSIAL (UJI T)")
print("-" * 50)
print("H0: bi = 0 (Variabel Xi tidak berpengaruh terhadap Y)")
print("H1: bi != 0 (Variabel Xi berpengaruh terhadap Y)")

t_tabel = stats.t.ppf(1 - ALPHA/2, df_resid)  # two-tailed test
print(f"\nT-tabel (alpha/2={ALPHA/2}, df={int(df_resid)}): +/-{t_tabel:.4f}")

print("\nHasil Uji Parsial:")
print("-" * 80)
print(f"{'Variabel':<25} {'Koefisien':>12} {'T-hitung':>12} {'P-value':>12} {'Keputusan':>15}")
print("-" * 80)

var_names = ['Konstanta', 'Pertumbuhan_Ekonomi', 'Tingkat_Pendidikan', 'Akses_Sanitasi']
for i, var in enumerate(var_names):
    coef = model_multi.params[i]
    t_stat = model_multi.tvalues[i]
    p_val = model_multi.pvalues[i]
    
    if p_val < ALPHA:
        keputusan = "SIGNIFIKAN"
    else:
        keputusan = "TIDAK SIGNIF."
    
    print(f"{var:<25} {coef:>12.4f} {t_stat:>12.4f} {p_val:>12.4f} {keputusan:>15}")

print("-" * 80)

# Interpretasi Uji Parsial
print("\nInterpretasi Uji Parsial:")
for i, var in enumerate(var_names[1:], 1):  # Skip konstanta
    coef = model_multi.params[i]
    p_val = model_multi.pvalues[i]
    t_stat = model_multi.tvalues[i]
    
    if p_val < ALPHA:
        pengaruh = "positif" if coef > 0 else "negatif"
        print(f"   - {var}: Berpengaruh {pengaruh} dan SIGNIFIKAN terhadap Stunting")
        print(f"     |t-hitung| ({abs(t_stat):.4f}) > t-tabel ({t_tabel:.4f}), p-value ({p_val:.4f}) < α ({ALPHA})")
    else:
        print(f"   - {var}: TIDAK SIGNIFIKAN terhadap Stunting")
        print(f"     |t-hitung| ({abs(t_stat):.4f}) <= t-tabel ({t_tabel:.4f}), p-value ({p_val:.4f}) >= α ({ALPHA})")

# Persamaan Regresi
print("\n" + "-" * 50)
print("PERSAMAAN REGRESI BERGANDA:")
print("-" * 50)
b0 = model_multi.params[0]
b1 = model_multi.params[1]
b2 = model_multi.params[2]
b3 = model_multi.params[3]

print(f"\nY = {b0:.4f} + ({b1:.4f})X1 + ({b2:.4f})X2 + ({b3:.4f})X3")
print("\nDimana:")
print("   Y  = Angka Stunting (prediksi)")
print("   X1 = Pertumbuhan Ekonomi")
print("   X2 = Tingkat Pendidikan")
print("   X3 = Akses Sanitasi")

# =====================================================
# 5.6 UJI ASUMSI KLASIK REGRESI
# =====================================================
print("\n" + "=" * 70)
print("5.6 UJI ASUMSI KLASIK REGRESI")
print("=" * 70)

# 5.6.1 Uji Multikolinearitas (VIF)
print("\n5.6.1 Uji Multikolinearitas (VIF - Variance Inflation Factor)")
print("-" * 50)
print("Kriteria: VIF < 10 = Tidak ada multikolinearitas")
print("          VIF >= 10 = Ada multikolinearitas")

X_vif = df[['Pertumbuhan_Ekonomi', 'Tingkat_Pendidikan', 'Akses_Sanitasi']]
X_vif = sm.add_constant(X_vif)

vif_data = pd.DataFrame()
vif_data['Variabel'] = X_vif.columns[1:]
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i+1) for i in range(len(X_vif.columns)-1)]

print("\nHasil Uji VIF:")
for idx, row in vif_data.iterrows():
    status = "OK (Tidak ada multikolinearitas)" if row['VIF'] < 10 else "MASALAH (Ada multikolinearitas)"
    print(f"   {row['Variabel']:25} : VIF = {row['VIF']:.4f} -> {status}")

# 5.6.2 Uji Normalitas Residual (Shapiro-Wilk)
print("\n5.6.2 Uji Normalitas Residual (Shapiro-Wilk)")
print("-" * 50)
print("H0: Residual berdistribusi normal")
print("H1: Residual tidak berdistribusi normal")

residuals = model_multi.resid
shapiro_stat, shapiro_pvalue = shapiro(residuals)

print(f"\nStatistik Shapiro-Wilk : {shapiro_stat:.4f}")
print(f"P-value                : {shapiro_pvalue:.4f}")
print(f"Keputusan (alpha={ALPHA}): {'Tolak H0 - Residual TIDAK normal' if shapiro_pvalue < ALPHA else 'Gagal Tolak H0 - Residual NORMAL'}")

# 5.6.3 Uji Heteroskedastisitas (Breusch-Pagan)
print("\n5.6.3 Uji Heteroskedastisitas (Breusch-Pagan)")
print("-" * 50)
print("H0: Tidak ada heteroskedastisitas (homoskedastis)")
print("H1: Ada heteroskedastisitas")

bp_test = het_breuschpagan(model_multi.resid, model_multi.model.exog)
bp_stat = bp_test[0]
bp_pvalue = bp_test[1]

print(f"\nStatistik Breusch-Pagan : {bp_stat:.4f}")
print(f"P-value                 : {bp_pvalue:.4f}")
print(f"Keputusan (alpha={ALPHA}): {'Tolak H0 - Ada HETEROSKEDASTISITAS' if bp_pvalue < ALPHA else 'Gagal Tolak H0 - HOMOSKEDASTIS (OK)'}")

# Ringkasan Asumsi Klasik
print("\n" + "-" * 50)
print("RINGKASAN UJI ASUMSI KLASIK:")
print("-" * 50)
vif_ok = all(vif_data['VIF'] < 10)
normal_ok = shapiro_pvalue >= ALPHA
homo_ok = bp_pvalue >= ALPHA

print(f"   1. Multikolinearitas  : {'TERPENUHI (VIF < 10)' if vif_ok else 'TIDAK TERPENUHI (VIF >= 10)'}")
print(f"   2. Normalitas Residual: {'TERPENUHI (p >= 0.05)' if normal_ok else 'TIDAK TERPENUHI (p < 0.05)'}")
print(f"   3. Homoskedastisitas  : {'TERPENUHI (p >= 0.05)' if homo_ok else 'TIDAK TERPENUHI (p < 0.05)'}")

if vif_ok and normal_ok and homo_ok:
    print("\n   >>> SEMUA ASUMSI KLASIK TERPENUHI - Model regresi VALID <<<")
else:
    print("\n   >>> PERHATIAN: Ada asumsi yang tidak terpenuhi <<<")

# =====================================================
# 6. VISUALISASI (MASING-MASING FILE TERPISAH)
# =====================================================
print("\n" + "=" * 70)
print("6. MEMBUAT VISUALISASI...")
print("=" * 70)

output_dir = 'D:/Semester 3/STATISTIKA INDUSTRI/statistika-industri/'

# 6.1 Scatter Plot dengan Garis Regresi - Pertumbuhan Ekonomi
fig1, ax1 = plt.subplots(figsize=(10, 7))
ax1.scatter(df['Pertumbuhan_Ekonomi'], df['Angka_Stunting'], 
            alpha=0.7, s=80, c='steelblue', edgecolors='white', linewidth=1)
z1 = np.polyfit(df['Pertumbuhan_Ekonomi'], df['Angka_Stunting'], 1)
p1 = np.poly1d(z1)
x_line1 = np.linspace(df['Pertumbuhan_Ekonomi'].min(), df['Pertumbuhan_Ekonomi'].max(), 100)
ax1.plot(x_line1, p1(x_line1), "r--", linewidth=2, 
         label=f'y = {z1[0]:.3f}x + {z1[1]:.3f}')
ax1.set_xlabel('Pertumbuhan Ekonomi (%)', fontsize=12)
ax1.set_ylabel('Angka Stunting (%)', fontsize=12)
ax1.set_title('Regresi Linear: Stunting vs Pertumbuhan Ekonomi', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(True, alpha=0.3)
r_squared1 = model1.rsquared
ax1.text(0.05, 0.95, f'R² = {r_squared1:.4f}\np-value = {model1.pvalues[1]:.4f}', 
         transform=ax1.transAxes, fontsize=11, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.savefig(output_dir + '1_regresi_pertumbuhan_ekonomi.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Disimpan: 1_regresi_pertumbuhan_ekonomi.png")

# 6.2 Scatter Plot dengan Garis Regresi - Tingkat Pendidikan
fig2, ax2 = plt.subplots(figsize=(10, 7))
ax2.scatter(df['Tingkat_Pendidikan'], df['Angka_Stunting'], 
            alpha=0.7, s=80, c='seagreen', edgecolors='white', linewidth=1)
z2 = np.polyfit(df['Tingkat_Pendidikan'], df['Angka_Stunting'], 1)
p2 = np.poly1d(z2)
x_line2 = np.linspace(df['Tingkat_Pendidikan'].min(), df['Tingkat_Pendidikan'].max(), 100)
ax2.plot(x_line2, p2(x_line2), "r--", linewidth=2, 
         label=f'y = {z2[0]:.3f}x + {z2[1]:.3f}')
ax2.set_xlabel('Tingkat Pendidikan (Tahun)', fontsize=12)
ax2.set_ylabel('Angka Stunting (%)', fontsize=12)
ax2.set_title('Regresi Linear: Stunting vs Tingkat Pendidikan', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=11)
ax2.grid(True, alpha=0.3)
r_squared2 = model2.rsquared
ax2.text(0.05, 0.95, f'R² = {r_squared2:.4f}\np-value = {model2.pvalues[1]:.4f}', 
         transform=ax2.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.savefig(output_dir + '2_regresi_tingkat_pendidikan.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Disimpan: 2_regresi_tingkat_pendidikan.png")

# 6.3 Scatter Plot dengan Garis Regresi - Akses Sanitasi
fig3, ax3 = plt.subplots(figsize=(10, 7))
ax3.scatter(df['Akses_Sanitasi'], df['Angka_Stunting'], 
            alpha=0.7, s=80, c='coral', edgecolors='white', linewidth=1)
z3 = np.polyfit(df['Akses_Sanitasi'], df['Angka_Stunting'], 1)
p3 = np.poly1d(z3)
x_line3 = np.linspace(df['Akses_Sanitasi'].min(), df['Akses_Sanitasi'].max(), 100)
ax3.plot(x_line3, p3(x_line3), "r--", linewidth=2, 
         label=f'y = {z3[0]:.3f}x + {z3[1]:.3f}')
ax3.set_xlabel('Akses Sanitasi Layak (%)', fontsize=12)
ax3.set_ylabel('Angka Stunting (%)', fontsize=12)
ax3.set_title('Regresi Linear: Stunting vs Akses Sanitasi', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right', fontsize=11)
ax3.grid(True, alpha=0.3)
r_squared3 = model3.rsquared
ax3.text(0.05, 0.95, f'R² = {r_squared3:.4f}\np-value = {model3.pvalues[1]:.4f}', 
         transform=ax3.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.savefig(output_dir + '3_regresi_akses_sanitasi.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Disimpan: 3_regresi_akses_sanitasi.png")

# 6.4 Heatmap Korelasi
fig4, ax4 = plt.subplots(figsize=(10, 8))
corr_matrix = df[['Angka_Stunting', 'Pertumbuhan_Ekonomi', 
                  'Tingkat_Pendidikan', 'Akses_Sanitasi']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
            fmt='.3f', linewidths=0.5, ax=ax4, vmin=-1, vmax=1,
            annot_kws={'size': 12})
ax4.set_title('Matriks Korelasi Antar Variabel', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir + '4_heatmap_korelasi.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Disimpan: 4_heatmap_korelasi.png")

# 6.5 Boxplot berdasarkan Kategori Stunting
fig5, ax5 = plt.subplots(figsize=(12, 7))
df_melted = pd.melt(df, id_vars=['Kategori_Stunting'], 
                     value_vars=['Pertumbuhan_Ekonomi', 'Tingkat_Pendidikan', 'Akses_Sanitasi'],
                     var_name='Variabel', value_name='Nilai')
for var in ['Pertumbuhan_Ekonomi', 'Tingkat_Pendidikan', 'Akses_Sanitasi']:
    mask = df_melted['Variabel'] == var
    min_val = df_melted.loc[mask, 'Nilai'].min()
    max_val = df_melted.loc[mask, 'Nilai'].max()
    df_melted.loc[mask, 'Nilai'] = (df_melted.loc[mask, 'Nilai'] - min_val) / (max_val - min_val)
sns.boxplot(x='Kategori_Stunting', y='Nilai', hue='Variabel', data=df_melted, ax=ax5)
ax5.set_title('Distribusi Variabel per Kategori Stunting (Normalized)', fontsize=14, fontweight='bold')
ax5.set_xlabel('Kategori Stunting', fontsize=12)
ax5.set_ylabel('Nilai (Normalized)', fontsize=12)
ax5.legend(title='Variabel', fontsize=10)
plt.tight_layout()
plt.savefig(output_dir + '5_boxplot_kategori_stunting.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Disimpan: 5_boxplot_kategori_stunting.png")

# 6.6 Bar Chart - Ringkasan Korelasi
fig6, ax6 = plt.subplots(figsize=(10, 7))
correlations = [
    corr_matrix.loc['Angka_Stunting', 'Pertumbuhan_Ekonomi'],
    corr_matrix.loc['Angka_Stunting', 'Tingkat_Pendidikan'],
    corr_matrix.loc['Angka_Stunting', 'Akses_Sanitasi']
]
variables = ['Pertumbuhan\nEkonomi', 'Tingkat\nPendidikan', 'Akses\nSanitasi']
colors = ['steelblue' if c >= 0 else 'coral' for c in correlations]
bars = ax6.bar(variables, correlations, color=colors, edgecolor='black', linewidth=1)
ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax6.set_ylabel('Koefisien Korelasi (r)', fontsize=12)
ax6.set_title('Korelasi dengan Angka Stunting', fontsize=14, fontweight='bold')
ax6.set_ylim(-1, 1)
for bar, corr in zip(bars, correlations):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{corr:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir + '6_bar_korelasi_stunting.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Disimpan: 6_bar_korelasi_stunting.png")

print("\n   >>> Semua 6 visualisasi telah disimpan sebagai file terpisah <<<")

# =====================================================
# 7. KESIMPULAN
# =====================================================
print("\n" + "=" * 70)
print("7. KESIMPULAN ANALISIS")
print("=" * 70)

print("\nKorelasi dengan Angka Stunting:")
print(f"   - Pertumbuhan Ekonomi : r = {corr_matrix.loc['Angka_Stunting', 'Pertumbuhan_Ekonomi']:.4f}")
print(f"   - Tingkat Pendidikan  : r = {corr_matrix.loc['Angka_Stunting', 'Tingkat_Pendidikan']:.4f}")
print(f"   - Akses Sanitasi      : r = {corr_matrix.loc['Angka_Stunting', 'Akses_Sanitasi']:.4f}")

print("\nR-squared Regresi Sederhana:")
print(f"   - Stunting ~ Pertumbuhan Ekonomi : R-squared = {model1.rsquared:.4f}")
print(f"   - Stunting ~ Tingkat Pendidikan  : R-squared = {model2.rsquared:.4f}")
print(f"   - Stunting ~ Akses Sanitasi      : R-squared = {model3.rsquared:.4f}")

print(f"\nR-squared Regresi Berganda: R-squared = {model_multi.rsquared:.4f}")
print(f"   Adjusted R-squared = {model_multi.rsquared_adj:.4f}")

print("\n" + "=" * 70)
print("ANALISIS SELESAI!")
print("=" * 70)
