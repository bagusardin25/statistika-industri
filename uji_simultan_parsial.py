import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# Data 34 Provinsi Indonesia Tahun 2023
data = {
    'Provinsi': ['Aceh', 'Sumatera Utara', 'Sumatera Barat', 'Riau', 'Jambi', 'Sumatera Selatan', 
                 'Bengkulu', 'Lampung', 'Kep. Bangka Belitung', 'Kepulauan Riau', 'DKI Jakarta', 
                 'Jawa Barat', 'Jawa Tengah', 'DI Yogyakarta', 'Jawa Timur', 'Banten', 'Bali', 
                 'Nusa Tenggara Barat', 'Nusa Tenggara Timur', 'Kalimantan Barat', 'Kalimantan Tengah', 
                 'Kalimantan Selatan', 'Kalimantan Timur', 'Kalimantan Utara', 'Sulawesi Utara', 
                 'Sulawesi Tengah', 'Sulawesi Selatan', 'Sulawesi Tenggara', 'Gorontalo', 'Sulawesi Barat', 
                 'Maluku', 'Maluku Utara', 'Papua', 'Papua Barat'],
    'Y': [29.4, 18.9, 23.6, 13.6, 13.5, 20.3, 20.2, 14.9, 20.6, 16.8, 17.6, 21.7, 20.7, 18.0, 17.7, 
          24.0, 7.2, 24.6, 37.9, 24.5, 23.5, 24.7, 22.9, 17.4, 21.3, 27.2, 27.4, 30.0, 26.9, 30.3, 
          28.4, 23.7, 28.6, 24.8],
    'X1': [4.23, 5.01, 4.62, 4.21, 4.67, 5.08, 4.28, 4.55, 4.38, 5.16, 4.96, 5.0, 4.97, 5.07, 4.95, 
           4.81, 5.71, 1.8, 3.47, 4.46, 4.14, 4.84, 6.22, 4.94, 5.48, 11.91, 4.51, 5.35, 4.5, 5.23, 
           5.21, 20.49, 5.18, 4.22],
    'X2': [9.55, 9.82, 9.28, 9.32, 8.81, 8.5, 9.03, 8.29, 8.25, 10.41, 11.45, 8.83, 8.01, 9.83, 8.11, 
           9.15, 9.45, 7.74, 7.82, 7.71, 8.73, 8.55, 9.99, 9.34, 9.77, 8.96, 8.76, 9.31, 8.1, 8.13, 
           10.2, 9.26, 7.15, 7.93],
    'X3': [78.85, 84.18, 70.97, 84.58, 83.04, 80.54, 80.28, 84.58, 93.21, 91.1, 93.5, 74.88, 85.2, 
           96.42, 83.72, 86.41, 95.7, 85.11, 75.67, 79.89, 76.31, 82.89, 91.21, 84.22, 85.91, 75.8, 
           93.69, 88.99, 81.72, 80.73, 78.17, 80.64, 43.0, 76.3]
}

df = pd.DataFrame(data)

# Variabel
Y = df['Y'].values  # Stunting
X1 = df['X1'].values  # Pertumbuhan Ekonomi
X2 = df['X2'].values  # Tingkat Pendidikan
X3 = df['X3'].values  # Akses Sanitasi

n = len(Y)  # Jumlah observasi (34 provinsi)
k = 3  # Jumlah variabel independen

alpha = 0.05

print("=" * 80)
print("ANALISIS PENGARUH PERTUMBUHAN EKONOMI, TINGKAT PENDIDIKAN, DAN AKSES SANITASI")
print("TERHADAP ANGKA STUNTING DI 34 PROVINSI INDONESIA TAHUN 2023")
print("=" * 80)

# Regresi Berganda menggunakan statsmodels
X = np.column_stack([X1, X2, X3])
X_with_const = sm.add_constant(X)
model = sm.OLS(Y, X_with_const).fit()

print("\n" + "=" * 80)
print("HASIL REGRESI BERGANDA")
print("=" * 80)
print(model.summary())

# ============================================================================
# RUMUS UJI SIMULTAN (F-TEST)
# ============================================================================
print("\n" + "=" * 80)
print("UJI SIMULTAN (UJI F)")
print("=" * 80)

print("""
RUMUS UJI SIMULTAN (F-TEST):
============================

        R^2 / k
F = -----------------------
    (1 - R^2) / (n - k - 1)

Atau:

        SSR / k              MSR
F = --------------------- = -----
    SSE / (n - k - 1)        MSE

Dimana:
- R^2 = Koefisien Determinasi
- k = Jumlah variabel independen (3)
- n = Jumlah observasi (34)
- SSR = Sum of Squares Regression (Jumlah Kuadrat Regresi)
- SSE = Sum of Squares Error (Jumlah Kuadrat Residual)
- MSR = Mean Square Regression
- MSE = Mean Square Error

Hipotesis:
- H0: b1 = b2 = b3 = 0 (Tidak ada pengaruh simultan)
- H1: Minimal satu bi != 0 (Ada pengaruh simultan)

Kriteria Keputusan:
- Tolak H0 jika F_hitung > F_tabel atau p-value < alpha (0.05)
""")

# Perhitungan manual
Y_mean = np.mean(Y)
Y_pred = model.predict(X_with_const)

SST = np.sum((Y - Y_mean)**2)  # Total Sum of Squares
SSR = np.sum((Y_pred - Y_mean)**2)  # Regression Sum of Squares
SSE = np.sum((Y - Y_pred)**2)  # Error Sum of Squares

R_squared = SSR / SST
MSR = SSR / k
MSE = SSE / (n - k - 1)
F_hitung = MSR / MSE

df1 = k  # Derajat bebas regresi
df2 = n - k - 1  # Derajat bebas residual
F_tabel = stats.f.ppf(1 - alpha, df1, df2)
p_value_F = 1 - stats.f.cdf(F_hitung, df1, df2)

print("PERHITUNGAN:")
print("-" * 50)
print(f"n (jumlah observasi)     = {n}")
print(f"k (variabel independen)  = {k}")
print(f"α (alpha)                = {alpha}")
print(f"")
print(f"SST (Total)              = {SST:.4f}")
print(f"SSR (Regression)         = {SSR:.4f}")
print(f"SSE (Error/Residual)     = {SSE:.4f}")
print(f"")
print(f"R²                       = {R_squared:.4f}")
print(f"Adjusted R²              = {model.rsquared_adj:.4f}")
print(f"")
print(f"MSR = SSR/k              = {SSR:.4f}/{k} = {MSR:.4f}")
print(f"MSE = SSE/(n-k-1)        = {SSE:.4f}/{n-k-1} = {MSE:.4f}")
print(f"")
print(f"F_hitung = MSR/MSE       = {MSR:.4f}/{MSE:.4f} = {F_hitung:.4f}")
print(f"F_tabel (α={alpha}, df1={df1}, df2={df2}) = {F_tabel:.4f}")
print(f"P-value                  = {p_value_F:.6f}")

print("\nKESIMPULAN UJI SIMULTAN:")
print("-" * 50)
if F_hitung > F_tabel:
    print(f"F_hitung ({F_hitung:.4f}) > F_tabel ({F_tabel:.4f})")
    print("Keputusan: TOLAK H0")
    print("Artinya: Pertumbuhan Ekonomi, Tingkat Pendidikan, dan Akses Sanitasi")
    print("         secara SIMULTAN berpengaruh signifikan terhadap Angka Stunting")
else:
    print(f"F_hitung ({F_hitung:.4f}) <= F_tabel ({F_tabel:.4f})")
    print("Keputusan: GAGAL TOLAK H0")
    print("Artinya: Pertumbuhan Ekonomi, Tingkat Pendidikan, dan Akses Sanitasi")
    print("         secara simultan TIDAK berpengaruh signifikan terhadap Angka Stunting")

# ============================================================================
# RUMUS UJI PARSIAL (T-TEST)
# ============================================================================
print("\n" + "=" * 80)
print("UJI PARSIAL (UJI T)")
print("=" * 80)

print("""
RUMUS UJI PARSIAL (T-TEST):
===========================

           bi
t = -------------
        SE(bi)

Dimana:
- bi = Koefisien regresi variabel ke-i
- SE(bi) = Standard Error koefisien ke-i

Hipotesis untuk setiap variabel:
- H0: bi = 0 (Tidak ada pengaruh parsial)
- H1: bi != 0 (Ada pengaruh parsial)

Kriteria Keputusan:
- Tolak H0 jika |t_hitung| > t_tabel atau p-value < alpha (0.05)
""")

# Ambil koefisien dan standard error
coefficients = model.params
std_errors = model.bse
t_values = model.tvalues
p_values = model.pvalues

t_tabel = stats.t.ppf(1 - alpha/2, n - k - 1)  # Two-tailed test

var_names = ['Konstanta (b0)', 'Pertumbuhan Ekonomi (b1)', 'Tingkat Pendidikan (b2)', 'Akses Sanitasi (b3)']

print("PERHITUNGAN:")
print("-" * 90)
print(f"t_tabel (α/2={alpha/2}, df={n-k-1}) = ±{t_tabel:.4f}")
print("-" * 90)
print(f"{'Variabel':<30} {'Koefisien':>12} {'Std Error':>12} {'t_hitung':>12} {'P-value':>12}")
print("-" * 90)

for i, var in enumerate(var_names):
    print(f"{var:<30} {coefficients[i]:>12.4f} {std_errors[i]:>12.4f} {t_values[i]:>12.4f} {p_values[i]:>12.6f}")

print("-" * 90)

print("\nKESIMPULAN UJI PARSIAL:")
print("-" * 90)

for i in range(1, len(var_names)):
    var = var_names[i]
    t_hit = abs(t_values[i])
    p_val = p_values[i]
    
    print(f"\n{var}:")
    print(f"  |t_hitung| = {t_hit:.4f}, t_tabel = {t_tabel:.4f}")
    print(f"  P-value = {p_val:.6f}")
    
    if t_hit > t_tabel:
        print(f"  Keputusan: TOLAK H0 (|t_hitung| > t_tabel)")
        print(f"  Artinya: {var.split('(')[0].strip()} berpengaruh SIGNIFIKAN terhadap Angka Stunting")
    else:
        print(f"  Keputusan: GAGAL TOLAK H0 (|t_hitung| <= t_tabel)")
        print(f"  Artinya: {var.split('(')[0].strip()} TIDAK berpengaruh signifikan terhadap Angka Stunting")

# ============================================================================
# PERSAMAAN REGRESI
# ============================================================================
print("\n" + "=" * 80)
print("PERSAMAAN REGRESI BERGANDA")
print("=" * 80)
print(f"""
Y = b0 + b1.X1 + b2.X2 + b3.X3

Y = {coefficients[0]:.4f} + ({coefficients[1]:.4f}).X1 + ({coefficients[2]:.4f}).X2 + ({coefficients[3]:.4f}).X3

Dimana:
- Y  = Prediksi Angka Stunting (%)
- X1 = Pertumbuhan Ekonomi (%)
- X2 = Tingkat Pendidikan (Rata-rata Lama Sekolah dalam tahun)
- X3 = Akses Sanitasi (%)
""")

# ============================================================================
# TABEL ANOVA
# ============================================================================
print("\n" + "=" * 80)
print("TABEL ANOVA")
print("=" * 80)
print(f"{'Sumber':<20} {'df':>8} {'Sum of Squares':>18} {'Mean Square':>15} {'F':>12} {'P-value':>12}")
print("-" * 85)
print(f"{'Regression':<20} {df1:>8} {SSR:>18.4f} {MSR:>15.4f} {F_hitung:>12.4f} {p_value_F:>12.6f}")
print(f"{'Residual':<20} {df2:>8} {SSE:>18.4f} {MSE:>15.4f}")
print(f"{'Total':<20} {n-1:>8} {SST:>18.4f}")
print("-" * 85)

print("\n" + "=" * 80)
print("RINGKASAN HASIL ANALISIS")
print("=" * 80)
print(f"""
1. Model Regresi: Y = {coefficients[0]:.4f} + ({coefficients[1]:.4f})X1 + ({coefficients[2]:.4f})X2 + ({coefficients[3]:.4f})X3

2. Koefisien Determinasi (R^2) = {R_squared:.4f} ({R_squared*100:.2f}%)
   Artinya: {R_squared*100:.2f}% variasi Angka Stunting dapat dijelaskan oleh 
   Pertumbuhan Ekonomi, Tingkat Pendidikan, dan Akses Sanitasi.

3. Uji Simultan (F-test):
   F_hitung = {F_hitung:.4f}, F_tabel = {F_tabel:.4f}, P-value = {p_value_F:.6f}
   Kesimpulan: {"Signifikan" if F_hitung > F_tabel else "Tidak Signifikan"} pada alpha = {alpha}

4. Uji Parsial (t-test):
   - Pertumbuhan Ekonomi (X1): t = {t_values[1]:.4f}, p = {p_values[1]:.6f} -> {"Signifikan" if abs(t_values[1]) > t_tabel else "Tidak Signifikan"}
   - Tingkat Pendidikan (X2): t = {t_values[2]:.4f}, p = {p_values[2]:.6f} -> {"Signifikan" if abs(t_values[2]) > t_tabel else "Tidak Signifikan"}
   - Akses Sanitasi (X3): t = {t_values[3]:.4f}, p = {p_values[3]:.6f} -> {"Signifikan" if abs(t_values[3]) > t_tabel else "Tidak Signifikan"}
""")
