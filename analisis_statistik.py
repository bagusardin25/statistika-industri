"""
ANALISIS PENGARUH PERTUMBUHAN EKONOMI DAN AKSES SANITASI 
TERHADAP ANGKA STUNTING DI 34 PROVINSI INDONESIA TAHUN 2023
================================================================
(Tanpa Variabel Tingkat Pendidikan)

Data: Pertumbuhan Ekonomi (X1) dan Akses Sanitasi (X2) terhadap Stunting (Y)

Rumus yang digunakan:
---------------------
1. Uji F (Simultan):
   F = (R² / k) / ((1 - R²) / (n - k - 1))
   
   Dimana:
   - R² = Koefisien determinasi
   - k  = Jumlah variabel independen
   - n  = Jumlah sampel
   
2. Uji t (Parsial):
   t = βi / SE(βi)
   
   Dimana:
   - βi     = Koefisien regresi variabel ke-i
   - SE(βi) = Standard error koefisien ke-i
"""

import numpy as np
import pandas as pd
from scipy import stats
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# DATA
# ============================================================
data_csv = """No,Provinsi,Stunting_Y,Ekonomi_X1,Pendidikan_X2,Sanitasi_X3
1,Aceh,29.4,4.23,9.55,78.85
2,Sumatera Utara,18.9,5.01,9.82,84.18
3,Sumatera Barat,23.6,4.62,9.28,70.97
4,Riau,13.6,4.21,9.32,84.58
5,Jambi,13.5,4.67,8.81,83.04
6,Sumatera Selatan,20.3,5.08,8.5,80.54
7,Bengkulu,20.2,4.28,9.03,80.28
8,Lampung,14.9,4.55,8.29,84.58
9,Kep. Bangka Belitung,20.6,4.38,8.25,93.21
10,Kepulauan Riau,16.8,5.16,10.41,91.1
11,DKI Jakarta,17.6,4.96,11.45,93.5
12,Jawa Barat,21.7,5.0,8.83,74.88
13,Jawa Tengah,20.7,4.97,8.01,85.2
14,DI Yogyakarta,18.0,5.07,9.83,96.42
15,Jawa Timur,17.7,4.95,8.11,83.72
16,Banten,24.0,4.81,9.15,86.41
17,Bali,7.2,5.71,9.45,95.7
18,Nusa Tenggara Barat,24.6,1.8,7.74,85.11
19,Nusa Tenggara Timur,37.9,3.47,7.82,75.67
20,Kalimantan Barat,24.5,4.46,7.71,79.89
21,Kalimantan Tengah,23.5,4.14,8.73,76.31
22,Kalimantan Selatan,24.7,4.84,8.55,82.89
23,Kalimantan Timur,22.9,6.22,9.99,91.21
24,Kalimantan Utara,17.4,4.94,9.34,84.22
25,Sulawesi Utara,21.3,5.48,9.77,85.91
26,Sulawesi Tengah,27.2,11.91,8.96,75.8
27,Sulawesi Selatan,27.4,4.51,8.76,93.69
28,Sulawesi Tenggara,30.0,5.35,9.31,88.99
29,Gorontalo,26.9,4.5,8.1,81.72
30,Sulawesi Barat,30.3,5.23,8.13,80.73
31,Maluku,28.4,5.21,10.2,78.17
32,Maluku Utara,23.7,20.49,9.26,80.64
33,Papua,28.6,5.18,7.15,43.0
34,Papua Barat,24.8,4.22,7.93,76.3"""

# Load data
df = pd.read_csv(StringIO(data_csv))

# Variabel
Y = df['Stunting_Y'].values  # Variabel Dependen
X1 = df['Ekonomi_X1'].values  # Variabel Independen 1 (Ekonomi)
# X2 = df['Pendidikan_X2'].values  # Variabel Independen 2 (DIHAPUS)
X3 = df['Sanitasi_X3'].values  # Variabel Independen 2 (Sanitasi)

# Matriks X dengan konstanta (intercept)
n = len(Y)  # Jumlah sampel
k = 2  # Jumlah variabel independen (Ekonomi dan Sanitasi)
X = np.column_stack([np.ones(n), X1, X3])  # Matriks desain [1, X1, X3]


# ============================================================
# FUNGSI PERHITUNGAN MANUAL
# ============================================================

def hitung_koefisien_regresi(X, Y):
    """
    Menghitung koefisien regresi menggunakan metode OLS (Ordinary Least Squares)
    Rumus: β = (X'X)^(-1) X'Y
    """
    XtX = np.dot(X.T, X)  # X transpose * X
    XtX_inv = np.linalg.inv(XtX)  # Inverse dari X'X
    XtY = np.dot(X.T, Y)  # X transpose * Y
    beta = np.dot(XtX_inv, XtY)  # Koefisien regresi
    return beta, XtX_inv


def hitung_r_squared(Y, Y_pred):
    """
    Menghitung R² (Koefisien Determinasi)
    Rumus: R² = SSR / SST = 1 - (SSE / SST)
    
    Dimana:
    - SST (Total Sum of Squares) = Σ(Yi - Ȳ)²
    - SSR (Regression Sum of Squares) = Σ(Ŷi - Ȳ)²
    - SSE (Error Sum of Squares) = Σ(Yi - Ŷi)²
    """
    Y_mean = np.mean(Y)
    SST = np.sum((Y - Y_mean) ** 2)  # Total Sum of Squares
    SSE = np.sum((Y - Y_pred) ** 2)  # Error Sum of Squares
    SSR = SST - SSE  # Regression Sum of Squares
    R_squared = SSR / SST
    return R_squared, SST, SSR, SSE


def uji_f_simultan(R_squared, n, k, SSR, SSE):
    """
    UJI F (SIMULTAN)
    ================
    Menguji signifikansi model regresi secara keseluruhan.
    
    Hipotesis:
    - H0: β1 = β2 = 0 (tidak ada pengaruh simultan)
    - H1: Minimal satu βi ≠ 0 (ada pengaruh simultan)
    
    Rumus F-hitung:
    F = (R² / k) / ((1 - R²) / (n - k - 1))
    
    atau secara equivalen:
    F = (SSR / k) / (SSE / (n - k - 1)) = MSR / MSE
    
    Dimana:
    - MSR = Mean Square Regression = SSR / k
    - MSE = Mean Square Error = SSE / (n - k - 1)
    - df1 = k (derajat kebebasan regresi)
    - df2 = n - k - 1 (derajat kebebasan error)
    """
    df1 = k  # Derajat kebebasan regresi
    df2 = n - k - 1  # Derajat kebebasan error
    
    # Rumus 1: Menggunakan R²
    F_hitung_v1 = (R_squared / k) / ((1 - R_squared) / (n - k - 1))
    
    # Rumus 2: Menggunakan SSR dan SSE
    MSR = SSR / df1  # Mean Square Regression
    MSE = SSE / df2  # Mean Square Error
    F_hitung_v2 = MSR / MSE
    
    # p-value
    p_value = 1 - stats.f.cdf(F_hitung_v1, df1, df2)
    
    # F-tabel (α = 0.05)
    alpha = 0.05
    F_tabel = stats.f.ppf(1 - alpha, df1, df2)
    
    return F_hitung_v1, F_tabel, p_value, df1, df2, MSR, MSE


def uji_t_parsial(beta, XtX_inv, MSE, n, k):
    """
    UJI t (PARSIAL)
    ===============
    Menguji signifikansi masing-masing koefisien regresi secara individu.
    
    Hipotesis (untuk setiap variabel):
    - H0: βi = 0 (variabel Xi tidak berpengaruh signifikan)
    - H1: βi ≠ 0 (variabel Xi berpengaruh signifikan)
    
    Rumus t-hitung:
    t = βi / SE(βi)
    
    Dimana:
    - SE(βi) = √(MSE × Cii)
    - Cii = Elemen diagonal ke-i dari matriks (X'X)^(-1)
    - MSE = Mean Square Error
    
    Derajat Kebebasan: df = n - k - 1
    """
    df = n - k - 1  # Derajat kebebasan
    
    # Standard Error untuk setiap koefisien
    # SE(βi) = √(MSE × diagonal(X'X)^(-1))
    var_beta = MSE * np.diag(XtX_inv)  # Variance dari koefisien
    SE_beta = np.sqrt(var_beta)  # Standard Error
    
    # t-hitung untuk setiap koefisien
    t_hitung = beta / SE_beta
    
    # p-value (two-tailed test)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_hitung), df))
    
    # t-tabel (α = 0.05, two-tailed)
    alpha = 0.05
    t_tabel = stats.t.ppf(1 - alpha/2, df)
    
    return t_hitung, t_tabel, p_values, SE_beta, df


# ============================================================
# EKSEKUSI ANALISIS
# ============================================================

print("=" * 80)
print("ANALISIS PENGARUH PERTUMBUHAN EKONOMI DAN AKSES SANITASI")
print("TERHADAP ANGKA STUNTING DI 34 PROVINSI INDONESIA TAHUN 2023")
print("(Tanpa Variabel Tingkat Pendidikan)")
print("=" * 80)

# 1. Hitung koefisien regresi
beta, XtX_inv = hitung_koefisien_regresi(X, Y)
Y_pred = np.dot(X, beta)  # Nilai prediksi

print("\n📊 PERSAMAAN REGRESI:")
print("-" * 60)
print(f"Y = {beta[0]:.4f} + ({beta[1]:.4f})X1 + ({beta[2]:.4f})X2")
print(f"\nDimana:")
print(f"  Y  = Angka Stunting (%)")
print(f"  X1 = Pertumbuhan Ekonomi  (β1 = {beta[1]:.4f})")
print(f"  X2 = Akses Sanitasi       (β2 = {beta[2]:.4f})")

# 2. Hitung R²
R_squared, SST, SSR, SSE = hitung_r_squared(Y, Y_pred)
R_squared_adj = 1 - ((1 - R_squared) * (n - 1) / (n - k - 1))

print(f"\n📈 KOEFISIEN DETERMINASI:")
print("-" * 50)
print(f"R²           = {R_squared:.4f} ({R_squared*100:.2f}%)")
print(f"R² Adjusted  = {R_squared_adj:.4f} ({R_squared_adj*100:.2f}%)")
print(f"\nArtinya: {R_squared*100:.2f}% variasi Angka Stunting dapat dijelaskan")
print(f"oleh variabel Pertumbuhan Ekonomi dan Akses Sanitasi.")

# 3. UJI F (SIMULTAN)
F_hitung, F_tabel, p_value_F, df1, df2, MSR, MSE = uji_f_simultan(R_squared, n, k, SSR, SSE)

print("\n" + "=" * 70)
print("📋 UJI SIMULTAN (UJI F)")
print("=" * 70)
print("""
RUMUS UJI F:
┌─────────────────────────────────────────────────────────────────┐
│                      R² / k                                     │
│  F-hitung = ─────────────────────────                           │
│              (1 - R²) / (n - k - 1)                             │
│                                                                 │
│  Atau equivalen:                                                │
│                                                                 │
│              MSR     SSR / k                                    │
│  F-hitung = ───── = ──────────────                              │
│              MSE     SSE / (n-k-1)                              │
└─────────────────────────────────────────────────────────────────┘

HIPOTESIS:
- H0: β1 = β2 = β3 = 0 (Tidak ada pengaruh simultan)
- H1: Minimal satu βi ≠ 0 (Ada pengaruh simultan)
""")

print(f"PERHITUNGAN:")
print("-" * 50)
print(f"n (jumlah sampel)       = {n}")
print(f"k (variabel independen) = {k}")
print(f"df1 (regresi)           = k = {df1}")
print(f"df2 (error)             = n - k - 1 = {df2}")
print(f"\nSSR (Sum of Squares Regression) = {SSR:.4f}")
print(f"SSE (Sum of Squares Error)      = {SSE:.4f}")
print(f"SST (Sum of Squares Total)      = {SST:.4f}")
print(f"\nMSR (Mean Square Regression)    = SSR/df1 = {MSR:.4f}")
print(f"MSE (Mean Square Error)         = SSE/df2 = {MSE:.4f}")

print(f"\nHASIL UJI F:")
print("-" * 50)
print(f"F-hitung  = {F_hitung:.4f}")
print(f"F-tabel   = {F_tabel:.4f} (α = 0.05, df1 = {df1}, df2 = {df2})")
print(f"p-value   = {p_value_F:.6f}")

print(f"\nKESIMPULAN:")
print("-" * 50)
if F_hitung > F_tabel:
    print(f"✅ F-hitung ({F_hitung:.4f}) > F-tabel ({F_tabel:.4f})")
    print(f"✅ p-value ({p_value_F:.6f}) < α (0.05)")
    print(f"\n🎯 KEPUTUSAN: TOLAK H0")
    print(f"   Artinya: Pertumbuhan Ekonomi dan Akses Sanitasi secara SIMULTAN")
    print(f"   berpengaruh signifikan terhadap Angka Stunting.")
else:
    print(f"❌ F-hitung ({F_hitung:.4f}) ≤ F-tabel ({F_tabel:.4f})")
    print(f"❌ p-value ({p_value_F:.6f}) ≥ α (0.05)")
    print(f"\n🎯 KEPUTUSAN: GAGAL TOLAK H0")
    print(f"   Artinya: Pertumbuhan Ekonomi dan Akses Sanitasi secara SIMULTAN")
    print(f"   TIDAK berpengaruh signifikan terhadap Angka Stunting.")

# 4. UJI t (PARSIAL)
t_hitung, t_tabel, p_values_t, SE_beta, df_t = uji_t_parsial(beta, XtX_inv, MSE, n, k)
var_names = ['Konstanta', 'Pertumbuhan Ekonomi (X1)', 'Akses Sanitasi (X2)']

print("\n" + "=" * 70)
print("📋 UJI PARSIAL (UJI t)")
print("=" * 70)
print("""
RUMUS UJI t:
┌─────────────────────────────────────────────────────────────────┐
│                βi                                               │
│  t-hitung = ────────                                            │
│              SE(βi)                                             │
│                                                                 │
│  Dimana:                                                        │
│  SE(βi) = √(MSE × Cii)                                          │
│  Cii    = Elemen diagonal ke-i dari matriks (X'X)⁻¹             │
└─────────────────────────────────────────────────────────────────┘

HIPOTESIS (untuk setiap variabel):
- H0: βi = 0 (Variabel Xi tidak berpengaruh signifikan)
- H1: βi ≠ 0 (Variabel Xi berpengaruh signifikan)

Derajat Kebebasan: df = n - k - 1 = """ + str(df_t) + """
t-tabel (α = 0.05, two-tailed) = """ + f"{t_tabel:.4f}" + """
""")

print("HASIL UJI t UNTUK SETIAP VARIABEL:")
print("-" * 70)
print(f"{'Variabel':<20} {'Koefisien':>12} {'SE':>12} {'t-hitung':>12} {'p-value':>12}")
print("-" * 70)

for i in range(len(beta)):
    print(f"{var_names[i]:<20} {beta[i]:>12.4f} {SE_beta[i]:>12.4f} {t_hitung[i]:>12.4f} {p_values_t[i]:>12.6f}")

print("\n" + "=" * 70)
print("KESIMPULAN UJI t (PARSIAL):")
print("=" * 70)

alpha = 0.05
for i in range(1, len(beta)):  # Skip konstanta
    print(f"\n📌 {var_names[i]}:")
    print(f"   t-hitung = {t_hitung[i]:.4f}")
    print(f"   t-tabel  = ±{t_tabel:.4f}")
    print(f"   p-value  = {p_values_t[i]:.6f}")
    
    if abs(t_hitung[i]) > t_tabel and p_values_t[i] < alpha:
        print(f"   ✅ |t-hitung| > t-tabel DAN p-value < 0.05")
        print(f"   🎯 KEPUTUSAN: TOLAK H0")
        print(f"   📊 Artinya: {var_names[i]} berpengaruh SIGNIFIKAN terhadap Stunting")
    else:
        print(f"   ❌ |t-hitung| ≤ t-tabel ATAU p-value ≥ 0.05")
        print(f"   🎯 KEPUTUSAN: GAGAL TOLAK H0")
        print(f"   📊 Artinya: {var_names[i]} TIDAK berpengaruh signifikan terhadap Stunting")

# 5. TABEL ANOVA
print("\n" + "=" * 70)
print("📋 TABEL ANOVA")
print("=" * 70)
print(f"\n{'Sumber Variasi':<20} {'df':>8} {'SS':>15} {'MS':>15} {'F':>12} {'p-value':>12}")
print("-" * 82)
print(f"{'Regresi':<20} {df1:>8} {SSR:>15.4f} {MSR:>15.4f} {F_hitung:>12.4f} {p_value_F:>12.6f}")
print(f"{'Error':<20} {df2:>8} {SSE:>15.4f} {MSE:>15.4f}")
print(f"{'Total':<20} {n-1:>8} {SST:>15.4f}")
print("-" * 82)

print("\n" + "=" * 70)
print("📋 RINGKASAN HASIL ANALISIS")
print("=" * 70)
print(f"""
┌─────────────────────────────────────────────────────────────────┐
│ Model Regresi (Tanpa Pendidikan):                               │
│ Stunting = {beta[0]:.2f} + ({beta[1]:.2f})Ekonomi + ({beta[2]:.2f})Sanitasi        │
│                                                                 │
│ R² = {R_squared:.4f} ({R_squared*100:.2f}% variasi dapat dijelaskan)           │
│                                                                 │
│ Uji F (Simultan):                                               │
│   F-hitung = {F_hitung:.4f} > F-tabel = {F_tabel:.4f}                       │
│   → Model {'SIGNIFIKAN' if F_hitung > F_tabel else 'TIDAK SIGNIFIKAN'}                                            │
│                                                                 │
│ Uji t (Parsial):                                                │
│   - Ekonomi:  t = {t_hitung[1]:>7.4f}, p = {p_values_t[1]:.4f}                       │
│   - Sanitasi: t = {t_hitung[2]:>7.4f}, p = {p_values_t[2]:.4f}                       │
└─────────────────────────────────────────────────────────────────┘
""")

print("=" * 70)
print("Analisis Selesai!")
print("=" * 70)

# ============================================================
# VISUALISASI
# ============================================================

print("\n🎨 Membuat visualisasi...")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.facecolor'] = 'white'

# Buat figure dengan 2x3 subplots
fig = plt.figure(figsize=(16, 12))
fig.suptitle('ANALISIS PENGARUH PERTUMBUHAN EKONOMI DAN AKSES SANITASI\nTERHADAP ANGKA STUNTING DI 34 PROVINSI INDONESIA TAHUN 2023', 
             fontsize=13, fontweight='bold', y=0.98)

# ============================================================
# 1. Scatter Plot: Ekonomi vs Stunting
# ============================================================
ax1 = fig.add_subplot(2, 3, 1)
ax1.scatter(X1, Y, color='#3498db', alpha=0.7, edgecolors='white', s=80, label='Data Aktual')

# Garis regresi sederhana untuk Ekonomi
z1 = np.polyfit(X1, Y, 1)
p1 = np.poly1d(z1)
x_line = np.linspace(X1.min(), X1.max(), 100)
ax1.plot(x_line, p1(x_line), color='#e74c3c', linewidth=2, linestyle='--', label=f'Regresi (β={z1[0]:.3f})')

ax1.set_xlabel('Pertumbuhan Ekonomi (X1)')
ax1.set_ylabel('Angka Stunting (Y)')
ax1.set_title('Pertumbuhan Ekonomi vs Stunting', fontweight='bold')
ax1.legend(loc='best', fontsize=8)
ax1.grid(True, alpha=0.3)

# ============================================================
# 2. Scatter Plot: Sanitasi vs Stunting
# ============================================================
ax2 = fig.add_subplot(2, 3, 2)
ax2.scatter(X3, Y, color='#2ecc71', alpha=0.7, edgecolors='white', s=80, label='Data Aktual')

# Garis regresi sederhana untuk Sanitasi
z2 = np.polyfit(X3, Y, 1)
p2 = np.poly1d(z2)
x_line2 = np.linspace(X3.min(), X3.max(), 100)
ax2.plot(x_line2, p2(x_line2), color='#e74c3c', linewidth=2, linestyle='--', label=f'Regresi (β={z2[0]:.3f})')

ax2.set_xlabel('Akses Sanitasi (X2)')
ax2.set_ylabel('Angka Stunting (Y)')
ax2.set_title('Akses Sanitasi vs Stunting', fontweight='bold')
ax2.legend(loc='best', fontsize=8)
ax2.grid(True, alpha=0.3)

# ============================================================
# 3. Nilai Aktual vs Prediksi
# ============================================================
ax3 = fig.add_subplot(2, 3, 3)
ax3.scatter(Y, Y_pred, color='#9b59b6', alpha=0.7, edgecolors='white', s=80)
ax3.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', linewidth=2, label='Garis Perfect Fit')
ax3.set_xlabel('Nilai Aktual (Y)')
ax3.set_ylabel('Nilai Prediksi (Ŷ)')
ax3.set_title(f'Aktual vs Prediksi (R² = {R_squared:.4f})', fontweight='bold')
ax3.legend(loc='best', fontsize=8)
ax3.grid(True, alpha=0.3)

# ============================================================
# 4. Residual Plot
# ============================================================
ax4 = fig.add_subplot(2, 3, 4)
residuals = Y - Y_pred
ax4.scatter(Y_pred, residuals, color='#e67e22', alpha=0.7, edgecolors='white', s=80)
ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Nilai Prediksi (Ŷ)')
ax4.set_ylabel('Residual (Y - Ŷ)')
ax4.set_title('Plot Residual', fontweight='bold')
ax4.grid(True, alpha=0.3)

# ============================================================
# 5. Perbandingan Koefisien (Bar Chart)
# ============================================================
ax5 = fig.add_subplot(2, 3, 5)
koef_names = ['Konstanta', 'Pertumbuhan\nEkonomi (X1)', 'Akses\nSanitasi (X2)']
koef_values = beta
colors = ['#95a5a6', '#3498db', '#2ecc71']

bars = ax5.bar(koef_names, koef_values, color=colors, edgecolor='white', linewidth=2)

# Tambahkan nilai di atas bar
for bar, val in zip(bars, koef_values):
    height = bar.get_height()
    ax5.annotate(f'{val:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax5.set_ylabel('Nilai Koefisien')
ax5.set_title('Koefisien Regresi', fontweight='bold')
ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax5.grid(True, alpha=0.3, axis='y')

# ============================================================
# 6. Hasil Uji Statistik (Visual Summary)
# ============================================================
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

# Buat tabel summary
summary_text = f"""
+==================================================+
|         RINGKASAN UJI STATISTIK                  |
+==================================================+
|                                                  |
|  [1] MODEL REGRESI:                              |
|  Y = {beta[0]:.2f} + ({beta[1]:.2f})X1 + ({beta[2]:.2f})X2           |
|                                                  |
|  [2] KOEFISIEN DETERMINASI:                      |
|  R^2 = {R_squared:.4f} ({R_squared*100:.2f}%)                       |
|  R^2 Adj = {R_squared_adj:.4f} ({R_squared_adj*100:.2f}%)                   |
|                                                  |
+==================================================+
|  [3] UJI F (SIMULTAN):                           |
|  F-hitung = {F_hitung:.4f}                               |
|  F-tabel  = {F_tabel:.4f}                               |
|  p-value  = {p_value_F:.6f}                           |
|  Hasil: {'[v] SIGNIFIKAN' if F_hitung > F_tabel else '[x] TIDAK SIGNIFIKAN'}                         |
|                                                  |
+==================================================+
|  [4] UJI t (PARSIAL):                            |
|                                                  |
|  Pertumbuhan Ekonomi (X1):                       |
|    t = {t_hitung[1]:.4f}, p = {p_values_t[1]:.4f}                    |
|    {'[x] Tidak Signifikan' if p_values_t[1] >= 0.05 else '[v] Signifikan'}                           |
|                                                  |
|  Akses Sanitasi (X2):                            |
|    t = {t_hitung[2]:.4f}, p = {p_values_t[2]:.4f}                    |
|    {'[x] Tidak Signifikan' if p_values_t[2] >= 0.05 else '[v] Signifikan'}                            |
|                                                  |
+==================================================+
"""

ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes,
         fontsize=9, verticalalignment='center', horizontalalignment='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6', linewidth=2))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Simpan gambar
output_path = 'visualisasi_analisis_regresi.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"\n✅ Visualisasi disimpan: {output_path}")

# Tampilkan gambar
plt.show()

print("\n" + "=" * 70)
print("Visualisasi Selesai!")
print("=" * 70)
