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
from scipy.stats import f_oneway, ttest_ind, ttest_1samp, levene
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
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
# 6. VISUALISASI
# =====================================================
print("\n" + "=" * 70)
print("6. MEMBUAT VISUALISASI...")
print("=" * 70)

# Membuat figure dengan multiple subplots
fig = plt.figure(figsize=(16, 20))

# 6.1 Scatter Plot dengan Garis Regresi - Pertumbuhan Ekonomi
ax1 = fig.add_subplot(3, 2, 1)
ax1.scatter(df['Pertumbuhan_Ekonomi'], df['Angka_Stunting'], 
            alpha=0.7, s=80, c='steelblue', edgecolors='white', linewidth=1)
z1 = np.polyfit(df['Pertumbuhan_Ekonomi'], df['Angka_Stunting'], 1)
p1 = np.poly1d(z1)
x_line1 = np.linspace(df['Pertumbuhan_Ekonomi'].min(), df['Pertumbuhan_Ekonomi'].max(), 100)
ax1.plot(x_line1, p1(x_line1), "r--", linewidth=2, 
         label=f'y = {z1[0]:.3f}x + {z1[1]:.3f}')
ax1.set_xlabel('Pertumbuhan Ekonomi (%)', fontsize=11)
ax1.set_ylabel('Angka Stunting (%)', fontsize=11)
ax1.set_title('Regresi Linear: Stunting vs Pertumbuhan Ekonomi', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Tambahkan R-squared
r_squared1 = model1.rsquared
ax1.text(0.05, 0.95, f'R² = {r_squared1:.4f}', transform=ax1.transAxes, 
         fontsize=10, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 6.2 Scatter Plot dengan Garis Regresi - Tingkat Pendidikan
ax2 = fig.add_subplot(3, 2, 2)
ax2.scatter(df['Tingkat_Pendidikan'], df['Angka_Stunting'], 
            alpha=0.7, s=80, c='seagreen', edgecolors='white', linewidth=1)
z2 = np.polyfit(df['Tingkat_Pendidikan'], df['Angka_Stunting'], 1)
p2 = np.poly1d(z2)
x_line2 = np.linspace(df['Tingkat_Pendidikan'].min(), df['Tingkat_Pendidikan'].max(), 100)
ax2.plot(x_line2, p2(x_line2), "r--", linewidth=2, 
         label=f'y = {z2[0]:.3f}x + {z2[1]:.3f}')
ax2.set_xlabel('Tingkat Pendidikan (Tahun)', fontsize=11)
ax2.set_ylabel('Angka Stunting (%)', fontsize=11)
ax2.set_title('Regresi Linear: Stunting vs Tingkat Pendidikan', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

r_squared2 = model2.rsquared
ax2.text(0.05, 0.95, f'R² = {r_squared2:.4f}', transform=ax2.transAxes, 
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 6.3 Scatter Plot dengan Garis Regresi - Akses Sanitasi
ax3 = fig.add_subplot(3, 2, 3)
ax3.scatter(df['Akses_Sanitasi'], df['Angka_Stunting'], 
            alpha=0.7, s=80, c='coral', edgecolors='white', linewidth=1)
z3 = np.polyfit(df['Akses_Sanitasi'], df['Angka_Stunting'], 1)
p3 = np.poly1d(z3)
x_line3 = np.linspace(df['Akses_Sanitasi'].min(), df['Akses_Sanitasi'].max(), 100)
ax3.plot(x_line3, p3(x_line3), "r--", linewidth=2, 
         label=f'y = {z3[0]:.3f}x + {z3[1]:.3f}')
ax3.set_xlabel('Akses Sanitasi Layak (%)', fontsize=11)
ax3.set_ylabel('Angka Stunting (%)', fontsize=11)
ax3.set_title('Regresi Linear: Stunting vs Akses Sanitasi', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

r_squared3 = model3.rsquared
ax3.text(0.05, 0.95, f'R² = {r_squared3:.4f}', transform=ax3.transAxes, 
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 6.4 Heatmap Korelasi
ax4 = fig.add_subplot(3, 2, 4)
corr_matrix = df[['Angka_Stunting', 'Pertumbuhan_Ekonomi', 
                  'Tingkat_Pendidikan', 'Akses_Sanitasi']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
            fmt='.3f', linewidths=0.5, ax=ax4, vmin=-1, vmax=1)
ax4.set_title('Matriks Korelasi', fontsize=12, fontweight='bold')

# 6.5 Boxplot berdasarkan Kategori Stunting
ax5 = fig.add_subplot(3, 2, 5)
df_melted = pd.melt(df, id_vars=['Kategori_Stunting'], 
                     value_vars=['Pertumbuhan_Ekonomi', 'Tingkat_Pendidikan', 'Akses_Sanitasi'],
                     var_name='Variabel', value_name='Nilai')

# Normalize untuk visualisasi yang lebih baik
for var in ['Pertumbuhan_Ekonomi', 'Tingkat_Pendidikan', 'Akses_Sanitasi']:
    mask = df_melted['Variabel'] == var
    # Normalisasi dengan min-max scaling
    min_val = df_melted.loc[mask, 'Nilai'].min()
    max_val = df_melted.loc[mask, 'Nilai'].max()
    df_melted.loc[mask, 'Nilai'] = (df_melted.loc[mask, 'Nilai'] - min_val) / (max_val - min_val)

sns.boxplot(x='Kategori_Stunting', y='Nilai', hue='Variabel', data=df_melted, ax=ax5)
ax5.set_title('Distribusi Variabel per Kategori Stunting (Normalized)', fontsize=12, fontweight='bold')
ax5.set_xlabel('Kategori Stunting', fontsize=11)
ax5.set_ylabel('Nilai (Normalized)', fontsize=11)
ax5.legend(title='Variabel', fontsize=9)

# 6.6 Bar Chart - Ringkasan Korelasi
ax6 = fig.add_subplot(3, 2, 6)
correlations = [
    corr_matrix.loc['Angka_Stunting', 'Pertumbuhan_Ekonomi'],
    corr_matrix.loc['Angka_Stunting', 'Tingkat_Pendidikan'],
    corr_matrix.loc['Angka_Stunting', 'Akses_Sanitasi']
]
variables = ['Pertumbuhan\nEkonomi', 'Tingkat\nPendidikan', 'Akses\nSanitasi']
colors = ['steelblue' if c >= 0 else 'coral' for c in correlations]

bars = ax6.bar(variables, correlations, color=colors, edgecolor='black', linewidth=1)
ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax6.set_ylabel('Koefisien Korelasi (r)', fontsize=11)
ax6.set_title('Korelasi dengan Angka Stunting', fontsize=12, fontweight='bold')
ax6.set_ylim(-1, 1)

# Tambahkan nilai di atas bar
for bar, corr in zip(bars, correlations):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{corr:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('D:/Semester 3/STATISTIKA INDUSTRI/statistika-industri/hasil_analisis_visualisasi.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi telah disimpan ke: hasil_analisis_visualisasi.png")

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
