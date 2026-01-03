import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

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

n = len(Y)  # Jumlah observasi (34 provinsi)
k = 2  # Jumlah variabel independen

alpha = 0.05

print("=" * 80)
print("ANALISIS PENGARUH PERTUMBUHAN EKONOMI DAN AKSES SANITASI")
print("TERHADAP ANGKA STUNTING DI 34 PROVINSI INDONESIA TAHUN 2023")
print("=" * 80)

# ============================================================================
# STATISTIK DESKRIPTIF
# ============================================================================
print("\n" + "=" * 80)
print("STATISTIK DESKRIPTIF")
print("=" * 80)

print("""
RUMUS STATISTIK DESKRIPTIF:
===========================

1. MEAN (Rata-rata):
                n
               Σ Xi
              i=1
   Mean = ----------
               n

2. MEDIAN (Nilai Tengah):
   - Jika n ganjil: Median = X[(n+1)/2]
   - Jika n genap : Median = (X[n/2] + X[(n/2)+1]) / 2

3. MODUS:
   Modus = Nilai dengan frekuensi tertinggi

4. STANDAR DEVIASI (Sampel):
              ___________________
             /   n
            /   Σ (Xi - X̄)²
           /   i=1
   s =    / ------------------
        \/        n - 1

5. VARIANS (Sampel):
              n
             Σ (Xi - X̄)²
            i=1
   s² = ----------------
             n - 1

6. MINIMUM: Nilai terkecil dalam data
7. MAXIMUM: Nilai terbesar dalam data
8. RANGE: Max - Min
""")

# Fungsi untuk menghitung modus
def hitung_modus(data):
    from collections import Counter
    rounded_data = [round(x, 1) for x in data]
    counter = Counter(rounded_data)
    max_freq = max(counter.values())
    if max_freq == 1:
        return "Tidak ada (semua nilai unik)", 1
    modes = [k for k, v in counter.items() if v == max_freq]
    return modes, max_freq

# Perhitungan untuk setiap variabel
variables = {
    'Y (Angka Stunting %)': Y,
    'X1 (Pertumbuhan Ekonomi %)': X1,
    'X2 (Akses Sanitasi %)': X2
}

print("PERHITUNGAN STATISTIK DESKRIPTIF:")
print("=" * 90)

for var_name, var_data in variables.items():
    print(f"\n>>> {var_name}")
    print("-" * 60)
    
    # Perhitungan manual
    n_data = len(var_data)
    sum_data = np.sum(var_data)
    mean_val = sum_data / n_data
    
    sorted_data = np.sort(var_data)
    if n_data % 2 == 0:
        median_val = (sorted_data[n_data//2 - 1] + sorted_data[n_data//2]) / 2
    else:
        median_val = sorted_data[n_data//2]
    
    modus_val, modus_freq = hitung_modus(var_data)
    
    sum_sq_diff = np.sum((var_data - mean_val)**2)
    variance_val = sum_sq_diff / (n_data - 1)
    std_val = np.sqrt(variance_val)
    
    min_val = np.min(var_data)
    max_val = np.max(var_data)
    range_val = max_val - min_val
    
    # Output perhitungan
    print(f"   n = {n_data}")
    print(f"   Σxi = {sum_data:.4f}")
    print(f"")
    print(f"   Mean    = Σxi / n = {sum_data:.4f} / {n_data} = {mean_val:.4f}")
    print(f"   Median  = {median_val:.4f}")
    print(f"   Modus   = {modus_val} (frekuensi: {modus_freq})")
    print(f"")
    print(f"   Σ(xi-x̄)² = {sum_sq_diff:.4f}")
    print(f"   Varians = Σ(xi-x̄)²/(n-1) = {sum_sq_diff:.4f}/{n_data-1} = {variance_val:.4f}")
    print(f"   Std Dev = √Varians = √{variance_val:.4f} = {std_val:.4f}")
    print(f"")
    print(f"   Min     = {min_val:.4f}")
    print(f"   Max     = {max_val:.4f}")
    print(f"   Range   = Max - Min = {max_val:.4f} - {min_val:.4f} = {range_val:.4f}")

# Tabel Ringkasan
print("\n" + "=" * 90)
print("TABEL RINGKASAN STATISTIK DESKRIPTIF")
print("=" * 90)
print(f"{'Statistik':<20} {'Y (Stunting)':>18} {'X1 (Ekonomi)':>18} {'X2 (Sanitasi)':>18}")
print("-" * 90)
print(f"{'N':<20} {n:>18} {n:>18} {n:>18}")
print(f"{'Mean':<20} {np.mean(Y):>18.4f} {np.mean(X1):>18.4f} {np.mean(X2):>18.4f}")
print(f"{'Median':<20} {np.median(Y):>18.4f} {np.median(X1):>18.4f} {np.median(X2):>18.4f}")
print(f"{'Std. Deviation':<20} {np.std(Y, ddof=1):>18.4f} {np.std(X1, ddof=1):>18.4f} {np.std(X2, ddof=1):>18.4f}")
print(f"{'Variance':<20} {np.var(Y, ddof=1):>18.4f} {np.var(X1, ddof=1):>18.4f} {np.var(X2, ddof=1):>18.4f}")
print(f"{'Minimum':<20} {np.min(Y):>18.4f} {np.min(X1):>18.4f} {np.min(X2):>18.4f}")
print(f"{'Maximum':<20} {np.max(Y):>18.4f} {np.max(X1):>18.4f} {np.max(X2):>18.4f}")
print(f"{'Range':<20} {np.max(Y)-np.min(Y):>18.4f} {np.max(X1)-np.min(X1):>18.4f} {np.max(X2)-np.min(X2):>18.4f}")
print("-" * 90)

# ============================================================================
# VISUALISASI STATISTIK DESKRIPTIF (MASING-MASING FILE TERPISAH)
# ============================================================================
print("\n" + "=" * 80)
print("MEMBUAT VISUALISASI STATISTIK DESKRIPTIF...")
print("=" * 80)

# 1. Histogram Y (Stunting) - File Terpisah
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.hist(Y, bins=8, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(np.mean(Y), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(Y):.2f}')
ax1.axvline(np.median(Y), color='green', linestyle='-.', linewidth=2, label=f'Median = {np.median(Y):.2f}')
ax1.set_xlabel('Angka Stunting (%)', fontsize=11)
ax1.set_ylabel('Frekuensi', fontsize=11)
ax1.set_title('Distribusi Y (Angka Stunting)\n34 Provinsi Indonesia Tahun 2023', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('D:/Semester 3/STATISTIKA INDUSTRI/statistika-industri/12a_histogram_Y_stunting.png', dpi=150, bbox_inches='tight')
plt.close()
print("Gambar 1: 12a_histogram_Y_stunting.png berhasil disimpan!")

# 2. Histogram X1 (Pertumbuhan Ekonomi) - File Terpisah
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.hist(X1, bins=8, color='forestgreen', edgecolor='black', alpha=0.7)
ax2.axvline(np.mean(X1), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(X1):.2f}')
ax2.axvline(np.median(X1), color='orange', linestyle='-.', linewidth=2, label=f'Median = {np.median(X1):.2f}')
ax2.set_xlabel('Pertumbuhan Ekonomi (%)', fontsize=11)
ax2.set_ylabel('Frekuensi', fontsize=11)
ax2.set_title('Distribusi X1 (Pertumbuhan Ekonomi)\n34 Provinsi Indonesia Tahun 2023', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('D:/Semester 3/STATISTIKA INDUSTRI/statistika-industri/12b_histogram_X1_ekonomi.png', dpi=150, bbox_inches='tight')
plt.close()
print("Gambar 2: 12b_histogram_X1_ekonomi.png berhasil disimpan!")

# 3. Histogram X2 (Akses Sanitasi) - File Terpisah
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.hist(X2, bins=8, color='coral', edgecolor='black', alpha=0.7)
ax3.axvline(np.mean(X2), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(X2):.2f}')
ax3.axvline(np.median(X2), color='blue', linestyle='-.', linewidth=2, label=f'Median = {np.median(X2):.2f}')
ax3.set_xlabel('Akses Sanitasi (%)', fontsize=11)
ax3.set_ylabel('Frekuensi', fontsize=11)
ax3.set_title('Distribusi X2 (Akses Sanitasi)\n34 Provinsi Indonesia Tahun 2023', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('D:/Semester 3/STATISTIKA INDUSTRI/statistika-industri/12c_histogram_X2_sanitasi.png', dpi=150, bbox_inches='tight')
plt.close()
print("Gambar 3: 12c_histogram_X2_sanitasi.png berhasil disimpan!")

# 4. Boxplot Semua Variabel - File Terpisah
fig4, ax4 = plt.subplots(figsize=(8, 6))
bp = ax4.boxplot([Y, X1, X2], tick_labels=['Y\n(Stunting)', 'X1\n(Ekonomi)', 'X2\n(Sanitasi)'],
                  patch_artist=True)
colors = ['steelblue', 'forestgreen', 'coral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_ylabel('Nilai (%)', fontsize=11)
ax4.set_title('Boxplot Perbandingan Variabel\n34 Provinsi Indonesia Tahun 2023', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('D:/Semester 3/STATISTIKA INDUSTRI/statistika-industri/12d_boxplot_perbandingan.png', dpi=150, bbox_inches='tight')
plt.close()
print("Gambar 4: 12d_boxplot_perbandingan.png berhasil disimpan!")

# 5. Bar Chart Mean, Median, Std Dev - File Terpisah
fig5, ax5 = plt.subplots(figsize=(10, 6))
stats_names = ['Mean', 'Median', 'Std Dev']
y_stats = [np.mean(Y), np.median(Y), np.std(Y, ddof=1)]
x1_stats = [np.mean(X1), np.median(X1), np.std(X1, ddof=1)]
x2_stats = [np.mean(X2), np.median(X2), np.std(X2, ddof=1)]

x_pos = np.arange(len(stats_names))
width = 0.25

bars1 = ax5.bar(x_pos - width, y_stats, width, label='Y (Stunting)', color='steelblue', alpha=0.7)
bars2 = ax5.bar(x_pos, x1_stats, width, label='X1 (Ekonomi)', color='forestgreen', alpha=0.7)
bars3 = ax5.bar(x_pos + width, x2_stats, width, label='X2 (Sanitasi)', color='coral', alpha=0.7)

ax5.set_ylabel('Nilai', fontsize=11)
ax5.set_title('Perbandingan Statistik Deskriptif\n34 Provinsi Indonesia Tahun 2023', fontsize=12, fontweight='bold')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(stats_names)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('D:/Semester 3/STATISTIKA INDUSTRI/statistika-industri/12e_barchart_statistik.png', dpi=150, bbox_inches='tight')
plt.close()
print("Gambar 5: 12e_barchart_statistik.png berhasil disimpan!")

# 6. Tabel Ringkasan Statistik Deskriptif - File Terpisah
fig6, ax6 = plt.subplots(figsize=(10, 8))
ax6.axis('off')

table_text = f"""
TABEL STATISTIK DESKRIPTIF
Pengaruh Pertumbuhan Ekonomi dan Akses Sanitasi terhadap Stunting
34 Provinsi Indonesia Tahun 2023

{'='*55}

{'Statistik':<18} {'Y (Stunting)':>12} {'X1 (Ekonomi)':>14} {'X2 (Sanitasi)':>14}
{'-'*55}
{'N':<18} {n:>12} {n:>14} {n:>14}
{'Mean':<18} {np.mean(Y):>12.4f} {np.mean(X1):>14.4f} {np.mean(X2):>14.4f}
{'Median':<18} {np.median(Y):>12.4f} {np.median(X1):>14.4f} {np.median(X2):>14.4f}
{'Std. Deviation':<18} {np.std(Y,ddof=1):>12.4f} {np.std(X1,ddof=1):>14.4f} {np.std(X2,ddof=1):>14.4f}
{'Variance':<18} {np.var(Y,ddof=1):>12.4f} {np.var(X1,ddof=1):>14.4f} {np.var(X2,ddof=1):>14.4f}
{'Minimum':<18} {np.min(Y):>12.4f} {np.min(X1):>14.4f} {np.min(X2):>14.4f}
{'Maximum':<18} {np.max(Y):>12.4f} {np.max(X1):>14.4f} {np.max(X2):>14.4f}
{'Range':<18} {np.max(Y)-np.min(Y):>12.4f} {np.max(X1)-np.min(X1):>14.4f} {np.max(X2)-np.min(X2):>14.4f}
{'-'*55}

Keterangan Variabel:
  Y  = Angka Stunting (%)
  X1 = Pertumbuhan Ekonomi (%)
  X2 = Akses Sanitasi (%)

{'='*55}
"""

ax6.text(0.5, 0.5, table_text, transform=ax6.transAxes, fontsize=11,
         verticalalignment='center', horizontalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
plt.tight_layout()
plt.savefig('D:/Semester 3/STATISTIKA INDUSTRI/statistika-industri/12f_tabel_statistik_deskriptif.png', dpi=150, bbox_inches='tight')
plt.close()
print("Gambar 6: 12f_tabel_statistik_deskriptif.png berhasil disimpan!")

# ============================================================================
# STATISTIK DESKRIPTIF VARIABEL KATEGORIK
# ============================================================================
print("\n" + "=" * 80)
print("STATISTIK DESKRIPTIF VARIABEL KATEGORIK")
print("=" * 80)

print("""
RUMUS STATISTIK DESKRIPTIF VARIABEL KATEGORIK:
==============================================

1. FREKUENSI (f):
   f = Jumlah kemunculan suatu kategori dalam data

2. FREKUENSI RELATIF (fr):
              f
   fr = ----------- x 100%
              n
   
   Dimana: f = frekuensi kategori, n = total data

3. FREKUENSI KUMULATIF (fk):
   fk = f1 + f2 + f3 + ... + fi
   
   (Penjumlahan frekuensi dari kategori pertama sampai kategori ke-i)

4. FREKUENSI KUMULATIF RELATIF (fkr):
              fk
   fkr = ----------- x 100%
              n
""")

# Membuat kategori dari variabel numerik untuk demonstrasi
# Kategorisasi Stunting berdasarkan WHO
def kategorisasi_stunting(nilai):
    if nilai < 20:
        return 'Rendah (<20%)'
    elif nilai < 30:
        return 'Sedang (20-30%)'
    else:
        return 'Tinggi (>=30%)'

# Kategorisasi Pertumbuhan Ekonomi
def kategorisasi_ekonomi(nilai):
    if nilai < 4:
        return 'Lambat (<4%)'
    elif nilai < 6:
        return 'Sedang (4-6%)'
    else:
        return 'Cepat (>=6%)'

# Kategorisasi Akses Sanitasi
def kategorisasi_sanitasi(nilai):
    if nilai < 75:
        return 'Rendah (<75%)'
    elif nilai < 85:
        return 'Sedang (75-85%)'
    else:
        return 'Baik (>=85%)'

# Membuat kolom kategorik
df['Y_Kategori'] = df['Y'].apply(kategorisasi_stunting)
df['X1_Kategori'] = df['X1'].apply(kategorisasi_ekonomi)
df['X2_Kategori'] = df['X2'].apply(kategorisasi_sanitasi)

# Fungsi untuk membuat tabel frekuensi lengkap
def buat_tabel_frekuensi(data, nama_variabel, urutan_kategori=None):
    print(f"\n>>> TABEL FREKUENSI: {nama_variabel}")
    print("-" * 75)
    
    # Hitung frekuensi
    freq = data.value_counts()
    if urutan_kategori:
        freq = freq.reindex(urutan_kategori).fillna(0).astype(int)
    
    n_total = len(data)
    
    # Hitung frekuensi relatif dan kumulatif
    freq_relatif = (freq / n_total) * 100
    freq_kumulatif = freq.cumsum()
    freq_kum_relatif = (freq_kumulatif / n_total) * 100
    
    # Tampilkan tabel
    print(f"{'Kategori':<25} {'Frekuensi (f)':>15} {'Fr. Relatif (%)':>18} {'Fk':>10} {'Fkr (%)':>12}")
    print("-" * 75)
    
    for kategori in freq.index:
        f = freq[kategori]
        fr = freq_relatif[kategori]
        fk = freq_kumulatif[kategori]
        fkr = freq_kum_relatif[kategori]
        print(f"{kategori:<25} {f:>15} {fr:>18.2f} {fk:>10} {fkr:>12.2f}")
    
    print("-" * 75)
    print(f"{'TOTAL':<25} {n_total:>15} {100.00:>18.2f}")
    
    return freq, freq_relatif, freq_kumulatif, freq_kum_relatif

# Tabel Frekuensi untuk setiap variabel kategorik
print("\nPERHITUNGAN TABEL FREKUENSI:")
print("=" * 75)

# Urutan kategori
urutan_stunting = ['Rendah (<20%)', 'Sedang (20-30%)', 'Tinggi (>=30%)']
urutan_ekonomi = ['Lambat (<4%)', 'Sedang (4-6%)', 'Cepat (>=6%)']
urutan_sanitasi = ['Rendah (<75%)', 'Sedang (75-85%)', 'Baik (>=85%)']

freq_Y, fr_Y, fk_Y, fkr_Y = buat_tabel_frekuensi(df['Y_Kategori'], 'Kategori Stunting (Y)', urutan_stunting)
freq_X1, fr_X1, fk_X1, fkr_X1 = buat_tabel_frekuensi(df['X1_Kategori'], 'Kategori Pertumbuhan Ekonomi (X1)', urutan_ekonomi)
freq_X2, fr_X2, fk_X2, fkr_X2 = buat_tabel_frekuensi(df['X2_Kategori'], 'Kategori Akses Sanitasi (X2)', urutan_sanitasi)

# Contoh perhitungan manual
print("\n" + "=" * 75)
print("CONTOH PERHITUNGAN MANUAL (Kategori Stunting):")
print("=" * 75)
print(f"""
Diketahui: n = {n} provinsi

Kategori 'Rendah (<20%)':
  - Frekuensi (f)           = {freq_Y.get('Rendah (<20%)', 0)} provinsi
  - Frekuensi Relatif (fr)  = f/n x 100% = {freq_Y.get('Rendah (<20%)', 0)}/{n} x 100% = {fr_Y.get('Rendah (<20%)', 0):.2f}%
  - Frekuensi Kumulatif     = {fk_Y.get('Rendah (<20%)', 0)}
  - Fk Relatif              = {fkr_Y.get('Rendah (<20%)', 0):.2f}%

Kategori 'Sedang (20-30%)':
  - Frekuensi (f)           = {freq_Y.get('Sedang (20-30%)', 0)} provinsi
  - Frekuensi Relatif (fr)  = f/n x 100% = {freq_Y.get('Sedang (20-30%)', 0)}/{n} x 100% = {fr_Y.get('Sedang (20-30%)', 0):.2f}%
  - Frekuensi Kumulatif     = {freq_Y.get('Rendah (<20%)', 0)} + {freq_Y.get('Sedang (20-30%)', 0)} = {fk_Y.get('Sedang (20-30%)', 0)}
  - Fk Relatif              = {fkr_Y.get('Sedang (20-30%)', 0):.2f}%

Kategori 'Tinggi (>=30%)':
  - Frekuensi (f)           = {freq_Y.get('Tinggi (>=30%)', 0)} provinsi
  - Frekuensi Relatif (fr)  = f/n x 100% = {freq_Y.get('Tinggi (>=30%)', 0)}/{n} x 100% = {fr_Y.get('Tinggi (>=30%)', 0):.2f}%
  - Frekuensi Kumulatif     = {fk_Y.get('Sedang (20-30%)', 0)} + {freq_Y.get('Tinggi (>=30%)', 0)} = {fk_Y.get('Tinggi (>=30%)', 0)}
  - Fk Relatif              = {fkr_Y.get('Tinggi (>=30%)', 0):.2f}%
""")

# ============================================================================
# VISUALISASI VARIABEL KATEGORIK
# ============================================================================
print("\n" + "=" * 80)
print("MEMBUAT VISUALISASI VARIABEL KATEGORIK...")
print("=" * 80)

# Bar Chart dan Pie Chart untuk Variabel Kategorik
fig_kat, axes_kat = plt.subplots(2, 3, figsize=(15, 10))

# Warna untuk setiap kategori
colors_stunting = ['#2ecc71', '#f1c40f', '#e74c3c']
colors_ekonomi = ['#e74c3c', '#f1c40f', '#2ecc71']
colors_sanitasi = ['#e74c3c', '#f1c40f', '#2ecc71']

# 1. Bar Chart Stunting
ax1 = axes_kat[0, 0]
freq_Y_ordered = freq_Y.reindex(urutan_stunting).fillna(0)
bars1 = ax1.bar(range(len(urutan_stunting)), freq_Y_ordered.values, color=colors_stunting, edgecolor='black')
ax1.set_xticks(range(len(urutan_stunting)))
ax1.set_xticklabels(['Rendah', 'Sedang', 'Tinggi'], fontsize=10)
ax1.set_ylabel('Frekuensi', fontsize=11)
ax1.set_title('Distribusi Kategori Stunting (Y)', fontsize=12, fontweight='bold')
for bar, val in zip(bars1, freq_Y_ordered.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{int(val)}', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# 2. Bar Chart Ekonomi
ax2 = axes_kat[0, 1]
freq_X1_ordered = freq_X1.reindex(urutan_ekonomi).fillna(0)
bars2 = ax2.bar(range(len(urutan_ekonomi)), freq_X1_ordered.values, color=colors_ekonomi, edgecolor='black')
ax2.set_xticks(range(len(urutan_ekonomi)))
ax2.set_xticklabels(['Lambat', 'Sedang', 'Cepat'], fontsize=10)
ax2.set_ylabel('Frekuensi', fontsize=11)
ax2.set_title('Distribusi Kategori Pertumbuhan Ekonomi (X1)', fontsize=12, fontweight='bold')
for bar, val in zip(bars2, freq_X1_ordered.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{int(val)}', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Bar Chart Sanitasi
ax3 = axes_kat[0, 2]
freq_X2_ordered = freq_X2.reindex(urutan_sanitasi).fillna(0)
bars3 = ax3.bar(range(len(urutan_sanitasi)), freq_X2_ordered.values, color=colors_sanitasi, edgecolor='black')
ax3.set_xticks(range(len(urutan_sanitasi)))
ax3.set_xticklabels(['Rendah', 'Sedang', 'Baik'], fontsize=10)
ax3.set_ylabel('Frekuensi', fontsize=11)
ax3.set_title('Distribusi Kategori Akses Sanitasi (X2)', fontsize=12, fontweight='bold')
for bar, val in zip(bars3, freq_X2_ordered.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{int(val)}', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Pie Chart Stunting
ax4 = axes_kat[1, 0]
sizes_Y = freq_Y_ordered.values
labels_Y = [f'{cat}\n({int(val)} provinsi)' for cat, val in zip(['Rendah', 'Sedang', 'Tinggi'], sizes_Y)]
ax4.pie(sizes_Y, labels=labels_Y, colors=colors_stunting, autopct='%1.1f%%', startangle=90, 
        explode=(0.02, 0.02, 0.02), shadow=True)
ax4.set_title('Proporsi Kategori Stunting (Y)', fontsize=12, fontweight='bold')

# 5. Pie Chart Ekonomi
ax5 = axes_kat[1, 1]
sizes_X1 = freq_X1_ordered.values
labels_X1 = [f'{cat}\n({int(val)} provinsi)' for cat, val in zip(['Lambat', 'Sedang', 'Cepat'], sizes_X1)]
ax5.pie(sizes_X1, labels=labels_X1, colors=colors_ekonomi, autopct='%1.1f%%', startangle=90,
        explode=(0.02, 0.02, 0.02), shadow=True)
ax5.set_title('Proporsi Kategori Pertumbuhan Ekonomi (X1)', fontsize=12, fontweight='bold')

# 6. Pie Chart Sanitasi
ax6 = axes_kat[1, 2]
sizes_X2 = freq_X2_ordered.values
labels_X2 = [f'{cat}\n({int(val)} provinsi)' for cat, val in zip(['Rendah', 'Sedang', 'Baik'], sizes_X2)]
ax6.pie(sizes_X2, labels=labels_X2, colors=colors_sanitasi, autopct='%1.1f%%', startangle=90,
        explode=(0.02, 0.02, 0.02), shadow=True)
ax6.set_title('Proporsi Kategori Akses Sanitasi (X2)', fontsize=12, fontweight='bold')

plt.suptitle('Statistik Deskriptif Variabel Kategorik\n34 Provinsi Indonesia Tahun 2023', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('D:/Semester 3/STATISTIKA INDUSTRI/statistika-industri/13_statistik_kategorik.png', dpi=150, bbox_inches='tight')
plt.close()
print("Gambar: 13_statistik_kategorik.png berhasil disimpan!")

# Tabel Frekuensi dalam bentuk gambar
fig_tabel, ax_tabel = plt.subplots(figsize=(12, 10))
ax_tabel.axis('off')

tabel_kategorik_text = f"""
TABEL FREKUENSI VARIABEL KATEGORIK
Pengaruh Pertumbuhan Ekonomi dan Akses Sanitasi terhadap Stunting
34 Provinsi Indonesia Tahun 2023

{'='*70}

1. TABEL FREKUENSI KATEGORI STUNTING (Y)
{'-'*70}
{'Kategori':<22} {'f':>8} {'fr (%)':>12} {'fk':>8} {'fkr (%)':>12}
{'-'*70}
{'Rendah (<20%)':<22} {int(freq_Y.get('Rendah (<20%)', 0)):>8} {fr_Y.get('Rendah (<20%)', 0):>12.2f} {int(fk_Y.get('Rendah (<20%)', 0)):>8} {fkr_Y.get('Rendah (<20%)', 0):>12.2f}
{'Sedang (20-30%)':<22} {int(freq_Y.get('Sedang (20-30%)', 0)):>8} {fr_Y.get('Sedang (20-30%)', 0):>12.2f} {int(fk_Y.get('Sedang (20-30%)', 0)):>8} {fkr_Y.get('Sedang (20-30%)', 0):>12.2f}
{'Tinggi (>=30%)':<22} {int(freq_Y.get('Tinggi (>=30%)', 0)):>8} {fr_Y.get('Tinggi (>=30%)', 0):>12.2f} {int(fk_Y.get('Tinggi (>=30%)', 0)):>8} {fkr_Y.get('Tinggi (>=30%)', 0):>12.2f}
{'-'*70}
{'TOTAL':<22} {n:>8} {100.00:>12.2f}

2. TABEL FREKUENSI KATEGORI PERTUMBUHAN EKONOMI (X1)
{'-'*70}
{'Kategori':<22} {'f':>8} {'fr (%)':>12} {'fk':>8} {'fkr (%)':>12}
{'-'*70}
{'Lambat (<4%)':<22} {int(freq_X1.get('Lambat (<4%)', 0)):>8} {fr_X1.get('Lambat (<4%)', 0):>12.2f} {int(fk_X1.get('Lambat (<4%)', 0)):>8} {fkr_X1.get('Lambat (<4%)', 0):>12.2f}
{'Sedang (4-6%)':<22} {int(freq_X1.get('Sedang (4-6%)', 0)):>8} {fr_X1.get('Sedang (4-6%)', 0):>12.2f} {int(fk_X1.get('Sedang (4-6%)', 0)):>8} {fkr_X1.get('Sedang (4-6%)', 0):>12.2f}
{'Cepat (>=6%)':<22} {int(freq_X1.get('Cepat (>=6%)', 0)):>8} {fr_X1.get('Cepat (>=6%)', 0):>12.2f} {int(fk_X1.get('Cepat (>=6%)', 0)):>8} {fkr_X1.get('Cepat (>=6%)', 0):>12.2f}
{'-'*70}
{'TOTAL':<22} {n:>8} {100.00:>12.2f}

3. TABEL FREKUENSI KATEGORI AKSES SANITASI (X2)
{'-'*70}
{'Kategori':<22} {'f':>8} {'fr (%)':>12} {'fk':>8} {'fkr (%)':>12}
{'-'*70}
{'Rendah (<75%)':<22} {int(freq_X2.get('Rendah (<75%)', 0)):>8} {fr_X2.get('Rendah (<75%)', 0):>12.2f} {int(fk_X2.get('Rendah (<75%)', 0)):>8} {fkr_X2.get('Rendah (<75%)', 0):>12.2f}
{'Sedang (75-85%)':<22} {int(freq_X2.get('Sedang (75-85%)', 0)):>8} {fr_X2.get('Sedang (75-85%)', 0):>12.2f} {int(fk_X2.get('Sedang (75-85%)', 0)):>8} {fkr_X2.get('Sedang (75-85%)', 0):>12.2f}
{'Baik (>=85%)':<22} {int(freq_X2.get('Baik (>=85%)', 0)):>8} {fr_X2.get('Baik (>=85%)', 0):>12.2f} {int(fk_X2.get('Baik (>=85%)', 0)):>8} {fkr_X2.get('Baik (>=85%)', 0):>12.2f}
{'-'*70}
{'TOTAL':<22} {n:>8} {100.00:>12.2f}

{'='*70}
Keterangan: f=frekuensi, fr=frekuensi relatif, fk=frekuensi kumulatif, fkr=frekuensi kumulatif relatif
"""

ax_tabel.text(0.5, 0.5, tabel_kategorik_text, transform=ax_tabel.transAxes, fontsize=10,
              verticalalignment='center', horizontalalignment='center',
              fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
plt.tight_layout()
plt.savefig('D:/Semester 3/STATISTIKA INDUSTRI/statistika-industri/14_tabel_frekuensi_kategorik.png', dpi=150, bbox_inches='tight')
plt.close()
print("Gambar: 14_tabel_frekuensi_kategorik.png berhasil disimpan!")

# Regresi Berganda menggunakan statsmodels
X = np.column_stack([X1, X2])
X_with_const = sm.add_constant(X)
model = sm.OLS(Y, X_with_const).fit()

print("\n" + "=" * 80)
print("HASIL REGRESI BERGANDA")
print("=" * 80)
print(model.summary())

# ============================================================================
# UJI SIMULTAN (F-TEST)
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

Hipotesis:
- H0: b1 = b2 = 0 (Tidak ada pengaruh simultan)
- H1: Minimal satu bi != 0 (Ada pengaruh simultan)

Kriteria Keputusan:
- Tolak H0 jika F_hitung > F_tabel atau p-value < alpha (0.05)
""")

# Perhitungan manual
Y_mean = np.mean(Y)
Y_pred = model.predict(X_with_const)

SST = np.sum((Y - Y_mean)**2)
SSR = np.sum((Y_pred - Y_mean)**2)
SSE = np.sum((Y - Y_pred)**2)

R_squared = SSR / SST
MSR = SSR / k
MSE = SSE / (n - k - 1)
F_hitung = MSR / MSE

df1 = k
df2 = n - k - 1
F_tabel = stats.f.ppf(1 - alpha, df1, df2)
p_value_F = 1 - stats.f.cdf(F_hitung, df1, df2)

print("PERHITUNGAN:")
print("-" * 50)
print(f"n (jumlah observasi)     = {n}")
print(f"k (variabel independen)  = {k}")
print(f"alpha                    = {alpha}")
print(f"")
print(f"SST (Total)              = {SST:.4f}")
print(f"SSR (Regression)         = {SSR:.4f}")
print(f"SSE (Error/Residual)     = {SSE:.4f}")
print(f"")
print(f"R^2                      = {R_squared:.4f}")
print(f"Adjusted R^2             = {model.rsquared_adj:.4f}")
print(f"")
print(f"MSR = SSR/k              = {SSR:.4f}/{k} = {MSR:.4f}")
print(f"MSE = SSE/(n-k-1)        = {SSE:.4f}/{n-k-1} = {MSE:.4f}")
print(f"")
print(f"F_hitung = MSR/MSE       = {MSR:.4f}/{MSE:.4f} = {F_hitung:.4f}")
print(f"F_tabel (alpha={alpha}, df1={df1}, df2={df2}) = {F_tabel:.4f}")
print(f"P-value                  = {p_value_F:.6f}")

print("\nKESIMPULAN UJI SIMULTAN:")
print("-" * 50)
if F_hitung > F_tabel:
    print(f"F_hitung ({F_hitung:.4f}) > F_tabel ({F_tabel:.4f})")
    print("Keputusan: TOLAK H0")
    print("Artinya: Pertumbuhan Ekonomi dan Akses Sanitasi secara SIMULTAN")
    print("         berpengaruh signifikan terhadap Angka Stunting")
else:
    print(f"F_hitung ({F_hitung:.4f}) <= F_tabel ({F_tabel:.4f})")
    print("Keputusan: GAGAL TOLAK H0")
    print("Artinya: Pertumbuhan Ekonomi dan Akses Sanitasi secara simultan")
    print("         TIDAK berpengaruh signifikan terhadap Angka Stunting")

# ============================================================================
# UJI PARSIAL (T-TEST)
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

Hipotesis untuk setiap variabel:
- H0: bi = 0 (Tidak ada pengaruh parsial)
- H1: bi != 0 (Ada pengaruh parsial)

Kriteria Keputusan:
- Tolak H0 jika |t_hitung| > t_tabel atau p-value < alpha (0.05)
""")

coefficients = model.params
std_errors = model.bse
t_values = model.tvalues
p_values = model.pvalues

t_tabel = stats.t.ppf(1 - alpha/2, n - k - 1)

var_names = ['Konstanta (b0)', 'Pertumbuhan Ekonomi (b1)', 'Akses Sanitasi (b2)']

print("PERHITUNGAN:")
print("-" * 90)
print(f"t_tabel (alpha/2={alpha/2}, df={n-k-1}) = +/-{t_tabel:.4f}")
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
Y = b0 + b1.X1 + b2.X2

Y = {coefficients[0]:.4f} + ({coefficients[1]:.4f}).X1 + ({coefficients[2]:.4f}).X2

Dimana:
- Y  = Prediksi Angka Stunting (%)
- X1 = Pertumbuhan Ekonomi (%)
- X2 = Akses Sanitasi (%)
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
1. Model Regresi: Y = {coefficients[0]:.4f} + ({coefficients[1]:.4f})X1 + ({coefficients[2]:.4f})X2

2. Koefisien Determinasi (R^2) = {R_squared:.4f} ({R_squared*100:.2f}%)
   Artinya: {R_squared*100:.2f}% variasi Angka Stunting dapat dijelaskan oleh 
   Pertumbuhan Ekonomi dan Akses Sanitasi.

3. Uji Simultan (F-test):
   F_hitung = {F_hitung:.4f}, F_tabel = {F_tabel:.4f}, P-value = {p_value_F:.6f}
   Kesimpulan: {"Signifikan" if F_hitung > F_tabel else "Tidak Signifikan"} pada alpha = {alpha}

4. Uji Parsial (t-test):
   - Pertumbuhan Ekonomi (X1): t = {t_values[1]:.4f}, p = {p_values[1]:.6f} -> {"Signifikan" if abs(t_values[1]) > t_tabel else "Tidak Signifikan"}
   - Akses Sanitasi (X2): t = {t_values[2]:.4f}, p = {p_values[2]:.6f} -> {"Signifikan" if abs(t_values[2]) > t_tabel else "Tidak Signifikan"}
""")

# ============================================================================
# VISUALISASI REGRESI LINIER
# ============================================================================
print("\n" + "=" * 80)
print("MEMBUAT VISUALISASI REGRESI LINIER...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Regresi Sederhana X1 (Pertumbuhan Ekonomi) vs Y (Stunting)
ax1 = axes[0, 0]
ax1.scatter(X1, Y, color='blue', alpha=0.7, edgecolors='black', s=80)
z1 = np.polyfit(X1, Y, 1)
p1 = np.poly1d(z1)
x1_line = np.linspace(X1.min(), X1.max(), 100)
ax1.plot(x1_line, p1(x1_line), color='red', linewidth=2, label=f'Y = {z1[1]:.2f} + {z1[0]:.2f}X1')
ax1.set_xlabel('Pertumbuhan Ekonomi (X1) %', fontsize=11)
ax1.set_ylabel('Angka Stunting (Y) %', fontsize=11)
ax1.set_title('Regresi Linier: Pertumbuhan Ekonomi vs Stunting', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 2. Regresi Sederhana X2 (Akses Sanitasi) vs Y (Stunting)
ax2 = axes[0, 1]
ax2.scatter(X2, Y, color='green', alpha=0.7, edgecolors='black', s=80)
z2 = np.polyfit(X2, Y, 1)
p2 = np.poly1d(z2)
x2_line = np.linspace(X2.min(), X2.max(), 100)
ax2.plot(x2_line, p2(x2_line), color='red', linewidth=2, label=f'Y = {z2[1]:.2f} + {z2[0]:.2f}X2')
ax2.set_xlabel('Akses Sanitasi (X2) %', fontsize=11)
ax2.set_ylabel('Angka Stunting (Y) %', fontsize=11)
ax2.set_title('Regresi Linier: Akses Sanitasi vs Stunting', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# 3. Actual vs Predicted (Regresi Berganda)
ax3 = axes[1, 0]
ax3.scatter(Y, Y_pred, color='purple', alpha=0.7, edgecolors='black', s=80)
min_val = min(Y.min(), Y_pred.min())
max_val = max(Y.max(), Y_pred.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Garis Ideal (Y=Y_pred)')
ax3.set_xlabel('Nilai Aktual Stunting (Y) %', fontsize=11)
ax3.set_ylabel('Nilai Prediksi Stunting %', fontsize=11)
ax3.set_title(f'Actual vs Predicted (R^2 = {R_squared:.4f})', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

# 4. Residual Plot
ax4 = axes[1, 1]
residuals = Y - Y_pred
ax4.scatter(Y_pred, residuals, color='orange', alpha=0.7, edgecolors='black', s=80)
ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Nilai Prediksi Stunting %', fontsize=11)
ax4.set_ylabel('Residual (Y - Y_pred)', fontsize=11)
ax4.set_title('Residual Plot', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.suptitle('Analisis Regresi: Pengaruh Pertumbuhan Ekonomi dan Akses Sanitasi\nterhadap Angka Stunting di 34 Provinsi Indonesia (2023)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('D:/Semester 3/STATISTIKA INDUSTRI/statistika-industri/10_regresi_linier_X1_X2.png', dpi=150, bbox_inches='tight')
plt.close()
print("Gambar 1: 10_regresi_linier_X1_X2.png berhasil disimpan!")

# ============================================================================
# VISUALISASI RINGKASAN UJI STATISTIK
# ============================================================================
fig2, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

summary_text = f"""
RINGKASAN HASIL ANALISIS REGRESI BERGANDA
Pengaruh Pertumbuhan Ekonomi (X1) dan Akses Sanitasi (X2) terhadap Stunting (Y)
34 Provinsi Indonesia Tahun 2023

{'='*70}

PERSAMAAN REGRESI:
Y = {coefficients[0]:.4f} + ({coefficients[1]:.4f})X1 + ({coefficients[2]:.4f})X2

{'='*70}

UJI SIMULTAN (F-TEST):
  - F_hitung = {F_hitung:.4f}
  - F_tabel  = {F_tabel:.4f}  (alpha=0.05, df1={df1}, df2={df2})
  - P-value  = {p_value_F:.6f}
  - Keputusan: {'TOLAK H0 - SIGNIFIKAN' if F_hitung > F_tabel else 'GAGAL TOLAK H0 - TIDAK SIGNIFIKAN'}

{'='*70}

UJI PARSIAL (T-TEST):  (t_tabel = {t_tabel:.4f}, alpha=0.05)

  Pertumbuhan Ekonomi (X1):
    - Koefisien = {coefficients[1]:.4f}
    - t_hitung  = {t_values[1]:.4f}
    - P-value   = {p_values[1]:.6f}
    - Keputusan: {'SIGNIFIKAN' if abs(t_values[1]) > t_tabel else 'TIDAK SIGNIFIKAN'}

  Akses Sanitasi (X2):
    - Koefisien = {coefficients[2]:.4f}
    - t_hitung  = {t_values[2]:.4f}
    - P-value   = {p_values[2]:.6f}
    - Keputusan: {'SIGNIFIKAN' if abs(t_values[2]) > t_tabel else 'TIDAK SIGNIFIKAN'}

{'='*70}

KOEFISIEN DETERMINASI:
  - R^2          = {R_squared:.4f} ({R_squared*100:.2f}%)
  - Adjusted R^2 = {model.rsquared_adj:.4f} ({model.rsquared_adj*100:.2f}%)
"""

ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='center', horizontalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('D:/Semester 3/STATISTIKA INDUSTRI/statistika-industri/11_ringkasan_uji_statistik.png', dpi=150, bbox_inches='tight')
plt.close()
print("Gambar 2: 11_ringkasan_uji_statistik.png berhasil disimpan!")

print("\nVisualisasi selesai dibuat!")
