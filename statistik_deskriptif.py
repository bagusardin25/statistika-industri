import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from scipy import stats
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
Y = df['Y'].values   # Stunting (%)
X1 = df['X1'].values # Pertumbuhan Ekonomi (%)
X2 = df['X2'].values # Akses Sanitasi (%)

n = len(Y)  # Jumlah observasi (34 provinsi)

print("=" * 90)
print("ANALISIS STATISTIK DESKRIPTIF")
print("Pengaruh Pertumbuhan Ekonomi dan Akses Sanitasi terhadap Angka Stunting")
print("34 Provinsi Indonesia Tahun 2023")
print("=" * 90)

# ============================================================================
# RUMUS STATISTIK DESKRIPTIF
# ============================================================================
print("\n" + "=" * 90)
print("RUMUS STATISTIK DESKRIPTIF")
print("=" * 90)

print("""
1. MEAN (RATA-RATA)
   ==================
   Rumus:
                 n
                 Σ Xi
                i=1
   Mean (X̄) = --------
                 n

   Keterangan:
   - Xi = Nilai data ke-i
   - n  = Jumlah data
   - Σ  = Simbol penjumlahan

   Interpretasi: Nilai rata-rata dari seluruh data observasi.

2. MEDIAN (NILAI TENGAH)
   ======================
   Rumus:
   - Jika n ganjil: Median = X[(n+1)/2]
   - Jika n genap : Median = (X[n/2] + X[(n/2)+1]) / 2

   Keterangan:
   - Data harus diurutkan terlebih dahulu dari kecil ke besar
   - n = Jumlah data

   Interpretasi: Nilai yang membagi data menjadi dua bagian sama besar.

3. MODUS (NILAI YANG PALING SERING MUNCUL)
   ========================================
   Rumus:
   Modus = Nilai dengan frekuensi tertinggi

   Keterangan:
   - Untuk data kontinu, modus dapat berupa kelas interval dengan frekuensi tertinggi
   - Data dapat memiliki lebih dari satu modus (bimodal, multimodal)

   Interpretasi: Nilai yang paling sering muncul dalam data.

4. STANDAR DEVIASI (SIMPANGAN BAKU)
   ==================================
   Rumus Sampel:
                    ___________________
                   /  n
                  /   Σ (Xi - X̄)²
                 /   i=1
   SD (s) =     / ------------------
              \/       n - 1

   Rumus Populasi:
                    ___________________
                   /  N
                  /   Σ (Xi - μ)²
                 /   i=1
   SD (σ) =     / ------------------
              \/        N

   Keterangan:
   - Xi = Nilai data ke-i
   - X̄  = Mean (rata-rata) sampel
   - μ  = Mean populasi
   - n  = Jumlah data sampel
   - N  = Jumlah data populasi

   Interpretasi: Ukuran sebaran data dari nilai rata-ratanya.

5. VARIANS
   ========
   Rumus Sampel:
                 n
                 Σ (Xi - X̄)²
                i=1
   Var (s²) = ---------------
                  n - 1

   Interpretasi: Kuadrat dari standar deviasi, mengukur variabilitas data.

6. MINIMUM
   ========
   Rumus:
   Min = X(1) = Nilai terkecil dalam data

   Interpretasi: Nilai terendah dari seluruh observasi.

7. MAXIMUM
   ========
   Rumus:
   Max = X(n) = Nilai terbesar dalam data

   Interpretasi: Nilai tertinggi dari seluruh observasi.

8. RANGE (JANGKAUAN)
   ==================
   Rumus:
   Range = Max - Min = X(n) - X(1)

   Interpretasi: Selisih antara nilai maksimum dan minimum.
""")

# ============================================================================
# FUNGSI PERHITUNGAN MANUAL
# ============================================================================

def hitung_mean(data):
    """Menghitung mean secara manual"""
    return sum(data) / len(data)

def hitung_median(data):
    """Menghitung median secara manual"""
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 1:  # Ganjil
        return sorted_data[n // 2]
    else:  # Genap
        return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2

def hitung_modus(data):
    """Menghitung modus secara manual"""
    from collections import Counter
    counter = Counter(data)
    max_freq = max(counter.values())
    modes = [k for k, v in counter.items() if v == max_freq]
    return modes, max_freq

def hitung_std_sampel(data):
    """Menghitung standar deviasi sampel secara manual"""
    mean = hitung_mean(data)
    n = len(data)
    sum_squared_diff = sum((x - mean) ** 2 for x in data)
    variance = sum_squared_diff / (n - 1)
    return np.sqrt(variance), variance

def hitung_min(data):
    """Menghitung nilai minimum"""
    return min(data)

def hitung_max(data):
    """Menghitung nilai maksimum"""
    return max(data)

def hitung_range(data):
    """Menghitung range"""
    return max(data) - min(data)

# ============================================================================
# PERHITUNGAN UNTUK SETIAP VARIABEL
# ============================================================================

variables = {
    'Y (Angka Stunting %)': Y,
    'X1 (Pertumbuhan Ekonomi %)': X1,
    'X2 (Akses Sanitasi %)': X2
}

print("\n" + "=" * 90)
print("PERHITUNGAN STATISTIK DESKRIPTIF")
print("=" * 90)

for var_name, var_data in variables.items():
    print(f"\n{'='*90}")
    print(f"VARIABEL: {var_name}")
    print(f"{'='*90}")
    
    # Perhitungan
    mean_val = hitung_mean(var_data)
    median_val = hitung_median(var_data)
    modus_val, modus_freq = hitung_modus(np.round(var_data, 1))
    std_val, var_val = hitung_std_sampel(var_data)
    min_val = hitung_min(var_data)
    max_val = hitung_max(var_data)
    range_val = hitung_range(var_data)
    
    print(f"\nJumlah Data (n) = {len(var_data)}")
    print(f"Jumlah Total (Σ) = {sum(var_data):.4f}")
    
    # Mean
    print(f"\n1. MEAN (Rata-rata):")
    print(f"   X̄ = Σxi / n")
    print(f"   X̄ = {sum(var_data):.4f} / {len(var_data)}")
    print(f"   X̄ = {mean_val:.4f}")
    
    # Median
    sorted_data = sorted(var_data)
    print(f"\n2. MEDIAN (Nilai Tengah):")
    print(f"   Data diurutkan: n = {len(var_data)} (genap)")
    print(f"   Posisi tengah: data ke-{len(var_data)//2} dan ke-{len(var_data)//2 + 1}")
    print(f"   Median = ({sorted_data[len(var_data)//2 - 1]:.2f} + {sorted_data[len(var_data)//2]:.2f}) / 2")
    print(f"   Median = {median_val:.4f}")
    
    # Modus
    print(f"\n3. MODUS (Nilai Paling Sering):")
    if modus_freq > 1:
        print(f"   Modus = {modus_val} (frekuensi: {modus_freq})")
    else:
        print(f"   Tidak ada modus (semua nilai unik/frekuensi = 1)")
    
    # Standar Deviasi
    print(f"\n4. STANDAR DEVIASI (Sampel):")
    print(f"   s = √[Σ(xi - x̄)² / (n-1)]")
    print(f"   Σ(xi - x̄)² = {sum((x - mean_val)**2 for x in var_data):.4f}")
    print(f"   s = √[{sum((x - mean_val)**2 for x in var_data):.4f} / {len(var_data)-1}]")
    print(f"   s = √{var_val:.4f}")
    print(f"   s = {std_val:.4f}")
    
    # Varians
    print(f"\n5. VARIANS (Sampel):")
    print(f"   s² = Σ(xi - x̄)² / (n-1)")
    print(f"   s² = {var_val:.4f}")
    
    # Min, Max, Range
    print(f"\n6. MINIMUM:")
    print(f"   Min = {min_val:.4f}")
    
    print(f"\n7. MAXIMUM:")
    print(f"   Max = {max_val:.4f}")
    
    print(f"\n8. RANGE (Jangkauan):")
    print(f"   Range = Max - Min")
    print(f"   Range = {max_val:.4f} - {min_val:.4f}")
    print(f"   Range = {range_val:.4f}")

# ============================================================================
# TABEL RINGKASAN STATISTIK DESKRIPTIF
# ============================================================================
print("\n" + "=" * 90)
print("TABEL RINGKASAN STATISTIK DESKRIPTIF")
print("=" * 90)

# Membuat tabel ringkasan
summary_data = {
    'Statistik': ['N (Jumlah Data)', 'Mean (Rata-rata)', 'Median', 'Modus', 
                  'Std. Deviation', 'Variance', 'Minimum', 'Maximum', 'Range'],
    'Y (Stunting)': [
        n,
        f"{np.mean(Y):.4f}",
        f"{np.median(Y):.4f}",
        'Tidak ada' if len(stats.mode(Y, keepdims=True).mode) == len(Y) else f"{stats.mode(Y, keepdims=True).mode[0]:.2f}",
        f"{np.std(Y, ddof=1):.4f}",
        f"{np.var(Y, ddof=1):.4f}",
        f"{np.min(Y):.4f}",
        f"{np.max(Y):.4f}",
        f"{np.max(Y) - np.min(Y):.4f}"
    ],
    'X1 (Ekonomi)': [
        n,
        f"{np.mean(X1):.4f}",
        f"{np.median(X1):.4f}",
        'Tidak ada' if len(stats.mode(X1, keepdims=True).mode) == len(X1) else f"{stats.mode(X1, keepdims=True).mode[0]:.2f}",
        f"{np.std(X1, ddof=1):.4f}",
        f"{np.var(X1, ddof=1):.4f}",
        f"{np.min(X1):.4f}",
        f"{np.max(X1):.4f}",
        f"{np.max(X1) - np.min(X1):.4f}"
    ],
    'X2 (Sanitasi)': [
        n,
        f"{np.mean(X2):.4f}",
        f"{np.median(X2):.4f}",
        'Tidak ada' if len(stats.mode(X2, keepdims=True).mode) == len(X2) else f"{stats.mode(X2, keepdims=True).mode[0]:.2f}",
        f"{np.std(X2, ddof=1):.4f}",
        f"{np.var(X2, ddof=1):.4f}",
        f"{np.min(X2):.4f}",
        f"{np.max(X2):.4f}",
        f"{np.max(X2) - np.min(X2):.4f}"
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n")
print(summary_df.to_string(index=False))

# ============================================================================
# VISUALISASI STATISTIK DESKRIPTIF
# ============================================================================
print("\n" + "=" * 90)
print("MEMBUAT VISUALISASI STATISTIK DESKRIPTIF...")
print("=" * 90)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Histogram Y (Stunting)
ax1 = axes[0, 0]
ax1.hist(Y, bins=10, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(np.mean(Y), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(Y):.2f}')
ax1.axvline(np.median(Y), color='green', linestyle='-.', linewidth=2, label=f'Median = {np.median(Y):.2f}')
ax1.set_xlabel('Angka Stunting (%)', fontsize=10)
ax1.set_ylabel('Frekuensi', fontsize=10)
ax1.set_title('Distribusi Y (Angka Stunting)', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Histogram X1 (Pertumbuhan Ekonomi)
ax2 = axes[0, 1]
ax2.hist(X1, bins=10, color='forestgreen', edgecolor='black', alpha=0.7)
ax2.axvline(np.mean(X1), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(X1):.2f}')
ax2.axvline(np.median(X1), color='orange', linestyle='-.', linewidth=2, label=f'Median = {np.median(X1):.2f}')
ax2.set_xlabel('Pertumbuhan Ekonomi (%)', fontsize=10)
ax2.set_ylabel('Frekuensi', fontsize=10)
ax2.set_title('Distribusi X1 (Pertumbuhan Ekonomi)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. Histogram X2 (Akses Sanitasi)
ax3 = axes[0, 2]
ax3.hist(X2, bins=10, color='coral', edgecolor='black', alpha=0.7)
ax3.axvline(np.mean(X2), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(X2):.2f}')
ax3.axvline(np.median(X2), color='blue', linestyle='-.', linewidth=2, label=f'Median = {np.median(X2):.2f}')
ax3.set_xlabel('Akses Sanitasi (%)', fontsize=10)
ax3.set_ylabel('Frekuensi', fontsize=10)
ax3.set_title('Distribusi X2 (Akses Sanitasi)', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Boxplot Semua Variabel (Normalized)
ax4 = axes[1, 0]
# Normalisasi untuk perbandingan
Y_norm = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
X1_norm = (X1 - np.min(X1)) / (np.max(X1) - np.min(X1))
X2_norm = (X2 - np.min(X2)) / (np.max(X2) - np.min(X2))
bp = ax4.boxplot([Y_norm, X1_norm, X2_norm], labels=['Y (Stunting)', 'X1 (Ekonomi)', 'X2 (Sanitasi)'],
                  patch_artist=True)
colors = ['steelblue', 'forestgreen', 'coral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_ylabel('Nilai Normalisasi (0-1)', fontsize=10)
ax4.set_title('Boxplot Perbandingan (Normalized)', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. Boxplot Y (Detail)
ax5 = axes[1, 1]
bp2 = ax5.boxplot([Y], labels=['Y (Stunting)'], patch_artist=True, widths=0.5)
bp2['boxes'][0].set_facecolor('steelblue')
bp2['boxes'][0].set_alpha(0.7)
ax5.set_ylabel('Angka Stunting (%)', fontsize=10)
ax5.set_title(f'Boxplot Y: Min={np.min(Y):.1f}, Max={np.max(Y):.1f}', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)
# Anotasi statistik
textstr = f'Mean: {np.mean(Y):.2f}\nMedian: {np.median(Y):.2f}\nStd: {np.std(Y, ddof=1):.2f}'
ax5.text(0.75, 0.95, textstr, transform=ax5.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 6. Bar Chart Ringkasan
ax6 = axes[1, 2]
stats_names = ['Mean', 'Median', 'Std Dev']
y_stats = [np.mean(Y), np.median(Y), np.std(Y, ddof=1)]
x1_stats = [np.mean(X1), np.median(X1), np.std(X1, ddof=1)]
x2_stats = [np.mean(X2), np.median(X2), np.std(X2, ddof=1)]

x = np.arange(len(stats_names))
width = 0.25

bars1 = ax6.bar(x - width, y_stats, width, label='Y (Stunting)', color='steelblue', alpha=0.7)
bars2 = ax6.bar(x, x1_stats, width, label='X1 (Ekonomi)', color='forestgreen', alpha=0.7)
bars3 = ax6.bar(x + width, x2_stats, width, label='X2 (Sanitasi)', color='coral', alpha=0.7)

ax6.set_ylabel('Nilai', fontsize=10)
ax6.set_title('Perbandingan Statistik Deskriptif', fontsize=11, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(stats_names)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

plt.suptitle('ANALISIS STATISTIK DESKRIPTIF\n34 Provinsi Indonesia Tahun 2023', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('D:/Semester 3/STATISTIKA INDUSTRI/statistika-industri/12_statistik_deskriptif.png', 
            dpi=150, bbox_inches='tight')
plt.close()
print("Gambar: 12_statistik_deskriptif.png berhasil disimpan!")

# ============================================================================
# INTERPRETASI HASIL
# ============================================================================
print("\n" + "=" * 90)
print("INTERPRETASI HASIL STATISTIK DESKRIPTIF")
print("=" * 90)

print(f"""
1. VARIABEL Y (ANGKA STUNTING):
   - Rata-rata stunting di 34 provinsi adalah {np.mean(Y):.2f}%
   - Nilai tengah (median) adalah {np.median(Y):.2f}%
   - Standar deviasi {np.std(Y, ddof=1):.2f}% menunjukkan variasi cukup besar antar provinsi
   - Provinsi dengan stunting terendah: {np.min(Y):.1f}% (Bali)
   - Provinsi dengan stunting tertinggi: {np.max(Y):.1f}% (Nusa Tenggara Timur)
   - Range {np.max(Y) - np.min(Y):.1f}% menunjukkan kesenjangan yang besar

2. VARIABEL X1 (PERTUMBUHAN EKONOMI):
   - Rata-rata pertumbuhan ekonomi adalah {np.mean(X1):.2f}%
   - Nilai tengah (median) adalah {np.median(X1):.2f}%
   - Standar deviasi {np.std(X1, ddof=1):.2f}% menunjukkan variasi yang tinggi
   - Pertumbuhan ekonomi terendah: {np.min(X1):.2f}% (NTB)
   - Pertumbuhan ekonomi tertinggi: {np.max(X1):.2f}% (Maluku Utara)

3. VARIABEL X2 (AKSES SANITASI):
   - Rata-rata akses sanitasi adalah {np.mean(X2):.2f}%
   - Nilai tengah (median) adalah {np.median(X2):.2f}%
   - Standar deviasi {np.std(X2, ddof=1):.2f}% menunjukkan variasi antar provinsi
   - Akses sanitasi terendah: {np.min(X2):.2f}% (Papua)
   - Akses sanitasi tertinggi: {np.max(X2):.2f}% (DI Yogyakarta)
""")

print("\n" + "=" * 90)
print("ANALISIS STATISTIK DESKRIPTIF SELESAI")
print("=" * 90)
