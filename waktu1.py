import time

# Inisialisasi map untuk menyimpan nilai waktu
time_map = {}

# Simpan waktu saat ini dalam map
time_map["start"] = time.time()

while True:
    now = time.time()

    # Ambil nilai waktu pertama kali dimasukkan
    first_entry_time = time_map.get("start")

    # Cek apakah waktu pertama kali dimasukkan <= waktu saat ini + 2 detik
    if first_entry_time+2 <= now:
        print("Waktu pertama kali dimasukkan kurang dari atau sama dengan waktu saat ini + 2 detik")
    else:
        print("tidak sama")

    # Keluar dari loop setelah 5 detik
    if now - first_entry_time >= 5:
        break
