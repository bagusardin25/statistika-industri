import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

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
    'X2': [78.85, 84.18, 70.97, 84.58, 83.04, 80.54, 80.28, 84.58, 93.21, 91.1, 93.5, 74.88, 85.2, 
           96.42, 83.72, 86.41, 95.7, 85.11, 75.67, 79.89, 76.31, 82.89, 91.21, 84.22, 85.91, 75.8, 
           93.69, 88.99, 81.72, 80.73, 78.17, 80.64, 43.0, 76.3]
}

df = pd.DataFrame(data)

# Variabel
Y = df['Y'].values   # Stunting
X1 = df['X1'].values # Pertumbuhan Ekonomi
X2 = df['X2'].values # Akses Sanitasi

# Konfigurasi output directory
output_dir = './'

print("=" * 80)
print("MEMBUAT VISUALISASI SCATTER PLOT")
print("=" * 80)

# =============================================================================
# SCATTER PLOT DENGAN GARIS REGRESI
# =============================================================================

# Membuat figure dengan 2 subplot sebaris
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# -------------------------------------------------------------------------------
# Scatter Plot 1: Pertumbuhan Ekonomi (X1) vs Stunting (Y)
# -------------------------------------------------------------------------------
ax1 = axes[0]

# Scatter points
scatter1 = ax1.scatter(X1, Y, c='#3498db', s=80, alpha=0.7, edgecolors='white', linewidth=1.5, label='Data Provinsi')

# Garis regresi linear
slope1, intercept1, r1, p1, se1 = stats.linregress(X1, Y)
x_line1 = np.linspace(X1.min(), X1.max(), 100)
y_line1 = slope1 * x_line1 + intercept1
ax1.plot(x_line1, y_line1, color='#e74c3c', linewidth=2.5, linestyle='-', label=f'Regresi Linear')

# Koefisien determinasi
r_squared1 = r1**2

# Styling
ax1.set_xlabel('Pertumbuhan Ekonomi X1 (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Angka Stunting Y (%)', fontsize=12, fontweight='bold')
ax1.set_title('Scatter Plot: Pertumbuhan Ekonomi vs Stunting\n34 Provinsi Indonesia Tahun 2023', 
              fontsize=13, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper right', fontsize=10)

# Anotasi persamaan regresi dan R²
text_box1 = f'Y = {intercept1:.2f} + ({slope1:.4f})X1\nR² = {r_squared1:.4f}\nr = {r1:.4f}'
ax1.text(0.05, 0.95, text_box1, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# -------------------------------------------------------------------------------
# Scatter Plot 2: Akses Sanitasi (X2) vs Stunting (Y)
# -------------------------------------------------------------------------------
ax2 = axes[1]

# Scatter points
scatter2 = ax2.scatter(X2, Y, c='#27ae60', s=80, alpha=0.7, edgecolors='white', linewidth=1.5, label='Data Provinsi')

# Garis regresi linear
slope2, intercept2, r2, p2, se2 = stats.linregress(X2, Y)
x_line2 = np.linspace(X2.min(), X2.max(), 100)
y_line2 = slope2 * x_line2 + intercept2
ax2.plot(x_line2, y_line2, color='#e74c3c', linewidth=2.5, linestyle='-', label=f'Regresi Linear')

# Koefisien determinasi
r_squared2 = r2**2

# Styling
ax2.set_xlabel('Akses Sanitasi X2 (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Angka Stunting Y (%)', fontsize=12, fontweight='bold')
ax2.set_title('Scatter Plot: Akses Sanitasi vs Stunting\n34 Provinsi Indonesia Tahun 2023', 
              fontsize=13, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper right', fontsize=10)

# Anotasi persamaan regresi dan R²
text_box2 = f'Y = {intercept2:.2f} + ({slope2:.4f})X2\nR² = {r_squared2:.4f}\nr = {r2:.4f}'
ax2.text(0.05, 0.95, text_box2, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig(output_dir + '17_scatter_plot_visualisasi.png', dpi=150, bbox_inches='tight')
plt.close()
print("Gambar: 17_scatter_plot_visualisasi.png berhasil disimpan!")

# =============================================================================
# SCATTER PLOT GABUNGAN DENGAN REGRESI BERGANDA
# =============================================================================

# Membuat figure untuk scatter plot dengan predicted values
fig2, ax3 = plt.subplots(figsize=(10, 7))

# Regresi berganda
X_multi = np.column_stack([X1, X2])
X_with_const = sm.add_constant(X_multi)
model = sm.OLS(Y, X_with_const).fit()
Y_pred = model.predict(X_with_const)

# Scatter Actual vs Predicted
scatter3 = ax3.scatter(Y_pred, Y, c='#9b59b6', s=100, alpha=0.7, edgecolors='white', linewidth=1.5, label='Data Observasi')

# Garis 45 derajat (perfect prediction line)
min_val = min(Y.min(), Y_pred.min())
max_val = max(Y.max(), Y_pred.max())
ax3.plot([min_val, max_val], [min_val, max_val], color='#e74c3c', linewidth=2.5, linestyle='--', label='Garis Perfect Fit (Y = Ŷ)')

# Styling
ax3.set_xlabel('Nilai Prediksi Ŷ (%)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Nilai Aktual Y (%)', fontsize=12, fontweight='bold')
ax3.set_title('Scatter Plot: Nilai Aktual vs Nilai Prediksi\nModel Regresi Berganda - 34 Provinsi Indonesia 2023', 
              fontsize=13, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(loc='upper left', fontsize=10)

# Anotasi model
r_squared_multi = model.rsquared
adj_r_squared = model.rsquared_adj
coefs = model.params
text_box3 = f'Model: Y = {coefs[0]:.2f} + ({coefs[1]:.4f})X1 + ({coefs[2]:.4f})X2\nR² = {r_squared_multi:.4f}\nAdj R² = {adj_r_squared:.4f}'
ax3.text(0.05, 0.95, text_box3, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

plt.tight_layout()
plt.savefig(output_dir + '18_scatter_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("Gambar: 18_scatter_actual_vs_predicted.png berhasil disimpan!")

# =============================================================================
# OUTPUT INFORMASI UNTUK INTERPRETASI
# =============================================================================
print("\n" + "=" * 80)
print("RINGKASAN UNTUK INTERPRETASI LAPORAN")
print("=" * 80)

print(f"""
SCATTER PLOT 1: PERTUMBUHAN EKONOMI (X1) vs STUNTING (Y)
---------------------------------------------------------
Persamaan Regresi : Y = {intercept1:.4f} + ({slope1:.4f})X1
Koefisien Korelasi (r) : {r1:.4f}
Koefisien Determinasi (R²) : {r_squared1:.4f}
Arah Hubungan : {'Negatif (berlawanan)' if slope1 < 0 else 'Positif (searah)'}

SCATTER PLOT 2: AKSES SANITASI (X2) vs STUNTING (Y)
---------------------------------------------------------
Persamaan Regresi : Y = {intercept2:.4f} + ({slope2:.4f})X2
Koefisien Korelasi (r) : {r2:.4f}
Koefisien Determinasi (R²) : {r_squared2:.4f}
Arah Hubungan : {'Negatif (berlawanan)' if slope2 < 0 else 'Positif (searah)'}

SCATTER PLOT 3: NILAI AKTUAL vs PREDIKSI (REGRESI BERGANDA)
---------------------------------------------------------
Persamaan Regresi Berganda : Y = {coefs[0]:.4f} + ({coefs[1]:.4f})X1 + ({coefs[2]:.4f})X2
Koefisien Determinasi (R²) : {r_squared_multi:.4f}
Adjusted R² : {adj_r_squared:.4f}
""")

print("=" * 80)
print("VISUALISASI SELESAI!")
print("=" * 80)
