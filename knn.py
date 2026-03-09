import os
import pickle
import numpy as np

# CIFAR-10 verisetinin bulunduğu klasör (Aynı dizinde bulunmalıdır)
dataset_path = "cifar-10-batches-py"

print("=====================================================")
print("  CIFAR-10 k-NN Sınıflandırıcı (Ödev 1) ")
print("=====================================================")

# 0. Veri Seti Kontrolü
if not os.path.exists(dataset_path):
    print(f"HATA: '{dataset_path}' klasörü bulunamadı.")
    print("Lütfen CIFAR-10 veri setini indirip dizine çıkararak çalıştırın.")
    exit()

print("\nAdım 1: Veri seti disk üzerinden yükleniyor (İnternetten çekilmiyor)...")

# Eğitim verilerini birleştirme (5 farklı batch var)
X_train_list = []
Y_train_list = []
for i in range(1, 6):
    file_path = os.path.join(dataset_path, f"data_batch_{i}")
    with open(file_path, 'rb') as f:
        dict_batch = pickle.load(f, encoding='bytes')
        X_train_list.append(dict_batch[b'data'])
        Y_train_list.append(dict_batch[b'labels'])

# Hesaplamaları vektörel ve hızlı yapmak için NumPy dizilerine çeviriyoruz
X_train = np.concatenate(X_train_list)
Y_train = np.concatenate(Y_train_list)

# Test verisini yükleme
test_file_path = os.path.join(dataset_path, "test_batch")
with open(test_file_path, 'rb') as f:
    dict_test = pickle.load(f, encoding='bytes')
    X_test = np.array(dict_test[b'data'])
    Y_test = np.array(dict_test[b'labels'])

# Sınıf (Label) isimlerini yükleme (örn: airplane, automobile, bird vs.)
meta_file_path = os.path.join(dataset_path, "batches.meta")
with open(meta_file_path, 'rb') as f:
    dict_meta = pickle.load(f, encoding='bytes')
    label_names = [label.decode('utf-8') for label in dict_meta[b'label_names']]

print(f"Eğitim verisi boyutu : {X_train.shape} (50000 resim, 3072 piksel/renk değeri)")
print(f"Test verisi boyutu   : {X_test.shape} (10000 resim)")

print("\n=====================================================")
print(" Adım 2: Kullanıcı Seçimleri")
print("=====================================================")

# 1. Mesafe metriği seçimi
while True:
    print("\nMesafe metriği seçiniz:")
    print("1 - L1 (Manhattan)")
    print("2 - L2 (Öklid)")
    uzaklik_secimi = input("Seçiminiz (1 veya 2): ").strip()
    if uzaklik_secimi in ['1', '2']:
        break
    print("Lütfen sadece '1' veya '2' giriniz.")

# 2. k değeri seçimi
while True:
    try:
        k_degeri = int(input("\nk değerini giriniz (Örn: 1, 3, 5): ").strip())
        if k_degeri > 0:
            break
        print("Lütfen pozitif bir tam sayı giriniz.")
    except ValueError:
        print("Lütfen geçerli bir sayı giriniz.")

# 3. Test edilecek nesne (Görsel) seçimi
while True:
    try:
        test_index = int(input(f"\nTest edilecek görselin indeksini giriniz (0 - {len(X_test)-1}): ").strip())
        if 0 <= test_index < len(X_test):
            break
        print("Sınırlar dışında bir indeks girdiniz.")
    except ValueError:
        print("Lütfen geçerli bir sayı giriniz.")


print("\n=====================================================")
print(" Adım 3: Sınıflandırma İşlemi")
print("=====================================================")

secilen_test_resmi = X_test[test_index]
gercek_etiket = Y_test[test_index]
gercek_sinif_adi = label_names[gercek_etiket]

print(f"Test edilen görselin GERÇEK sınıfı: {gercek_sinif_adi.upper()}")
print("Mesafe hesaplanıyor...")

# Tüm eğitim verisi ile tek tek distance hesaplamak (Açık ve düz kod yazımı için loop yerine numpy matris işlemi kullanıldı çünkü 50,000 resim için klasik döngü çok uzun sürer)
if uzaklik_secimi == '1':
    # L1 (Manhattan) Uzaklığı = Mutlak Farkların Toplamı
    fark_matrisi = np.abs(X_train - secilen_test_resmi)
    mesafeler = np.sum(fark_matrisi, axis=1)
    metrik_adi = "L1 (Manhattan)"

elif uzaklik_secimi == '2':
    # L2 (Öklid) Uzaklığı = Farkların Karesinin Toplamının Karekökü
    fark_matrisi = np.square(X_train - secilen_test_resmi)
    mesafeler = np.sqrt(np.sum(fark_matrisi, axis=1))
    metrik_adi = "L2 (Öklid)"


# Mesafeleri sıralayıp en yakın 'k' tanesinin indekslerini alıyoruz
# argsort mesafeleri küçükten büyüğe sıralar ve o mesafelerin orjinal dizideki indekslerini döndürür.
sirali_indeksler = np.argsort(mesafeler)

print(f"\nSeçilen Metrik: {metrik_adi}")
print(f"En yakın {k_degeri} komşu bulunuyor...\n")

# En yakın k eğitim verisinin etiketlerini topluyoruz
en_yakin_k_etiketler = []
for i in range(k_degeri):
    komsu_indeks = sirali_indeksler[i]
    komsu_etiket = Y_train[komsu_indeks]
    komsu_mesafe = mesafeler[komsu_indeks]
    komsu_sinif_adi = label_names[komsu_etiket]
    
    en_yakin_k_etiketler.append(komsu_etiket)
    print(f"{i+1}. En Yakın Komşu -> Mesafe: {komsu_mesafe:.2f} | Sınıf: {komsu_sinif_adi}")

# Sınıf sayımı yapıyoruz (Oylama - Majority Voting)
oy_sayilari = {}
for etiket in en_yakin_k_etiketler:
    if etiket in oy_sayilari:
        oy_sayilari[etiket] += 1
    else:
        oy_sayilari[etiket] = 1

# En çok oy alan sınıfı bulma
tahmin_edilen_etiket = -1
en_yuksek_oy = -1

for etiket, oy in oy_sayilari.items():
    if oy > en_yuksek_oy:
        en_yuksek_oy = oy
        tahmin_edilen_etiket = etiket

tahmin_sinif_adi = label_names[tahmin_edilen_etiket]

print("\n=====================================================")
print(" SONUÇ")
print("=====================================================")
print(f"Gerçek Sınıf   : {gercek_sinif_adi}")
print(f"Tahmin Edilen  : {tahmin_sinif_adi} ({en_yuksek_oy}/{k_degeri} oy)")

if tahmin_edilen_etiket == gercek_etiket:
    print("\n-> Sınıflandırma BAŞARILI!")
else:
    print("\n-> Sınıflandırma HATALI!")
