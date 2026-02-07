# diploma_kmeans_research.py
# Дослідження стиснення зображень методом кластеризації K-means
# Метрики: CR (теоретичний і фактичний), PSNR, SSIM
# Оптимальне K: Elbow method, Silhouette Coefficient
# Дослідження стійкості до шуму: AWGN (білий гаусівський) sigma = 5,10,15,20,30
#
# Запуск:
#   python diploma_kmeans_research.py --input-dir images --output-dir out --ks 4,8,16,32,64
#
# Залежності:
#   pip install numpy pillow scikit-learn scikit-image matplotlib

from __future__ import annotations
import argparse
import csv
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from skimage.metrics import structural_similarity as ssim_metric

import matplotlib.pyplot as plt


# -----------------------------
# Утиліти вводу/виводу
# -----------------------------

def load_rgb(path: str) -> np.ndarray:
    """Завантаження зображення як RGB uint8 (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def save_rgb(arr: np.ndarray, path: str) -> None:
    """Збереження RGB uint8 (H, W, 3)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def file_size_bytes(path: str) -> int:
    return int(os.path.getsize(path))


# -----------------------------
# Метрики якості
# -----------------------------

def mse_psnr(orig: np.ndarray, comp: np.ndarray) -> Tuple[float, float]:
    """MSE і PSNR (дБ) для 8-бітного RGB."""
    o = orig.astype(np.float32)
    c = comp.astype(np.float32)
    mse = float(np.mean((o - c) ** 2))
    if mse == 0.0:
        return mse, float("inf")
    max_i = 255.0
    psnr = float(20.0 * np.log10(max_i) - 10.0 * np.log10(mse))
    return mse, psnr


def ssim_rgb(orig: np.ndarray, comp: np.ndarray) -> float:
    """SSIM для RGB (враховує візуальну якість)."""
    return float(ssim_metric(orig, comp, channel_axis=2, data_range=255))


# -----------------------------
# Шум
# -----------------------------

def add_awgn(img: np.ndarray, sigma: float, seed: int = 42) -> np.ndarray:
    """
    Адитивний білий гаусівський шум N(0, sigma^2) для RGB.
    sigma у шкалі 0..255.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
    return noisy


# -----------------------------
# Теоретичний CR
# -----------------------------

def compression_ratio_theoretical(h: int, w: int, k: int) -> float:
    """
    Теоретичний CR без урахування конкретного формату файлу:
    Оригінал: 24 біт/піксель.
    Після квантування:
      - індекс кластера: ceil(log2(K)) біт/піксель
      - палітра: K*24 біт
    CR = original_bits / compressed_bits
    """
    n = h * w
    original_bits = n * 24
    index_bits = int(np.ceil(np.log2(max(k, 2))))
    compressed_bits = n * index_bits + k * 24
    return float(original_bits / compressed_bits)


# -----------------------------
# K-means стиснення
# -----------------------------

def kmeans_compress(
    img: np.ndarray,
    k: int,
    seed: int = 42,
    max_iter: int = 300,
    n_init: int = 10,
    sample_pixels: Optional[int] = 50000,
) -> Tuple[np.ndarray, float]:
    """
    Стиснення через K-means (квантування кольорів).
    Для швидкості KMeans навчається на підвибірці пікселів (sample_pixels),
    але центри застосовуються до всіх пікселів.
    Повертає: (стиснене зображення, inertia)
    """
    h, w, c = img.shape
    if c != 3:
        raise ValueError("Очікується RGB (3 канали).")
    if k < 2:
        raise ValueError("K має бути >= 2.")

    pixels = img.reshape(-1, 3).astype(np.float32)
    n = pixels.shape[0]

    if sample_pixels is not None and 0 < sample_pixels < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample_pixels, replace=False)
        train_pixels = pixels[idx]
    else:
        train_pixels = pixels

    model = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
        algorithm="lloyd",
    )
    model.fit(train_pixels)

    labels = model.predict(pixels)
    centers = model.cluster_centers_  # (K, 3)

    comp_pixels = centers[labels]
    comp = comp_pixels.reshape(h, w, 3)
    comp = np.clip(np.rint(comp), 0, 255).astype(np.uint8)

    return comp, float(model.inertia_)


# -----------------------------
# Оптимальне K: Elbow і Silhouette
# -----------------------------

def choose_k_elbow(ks: List[int], inertias: List[float]) -> int:
    """
    Автовибір K методом ліктя:
    максимальна перпендикулярна відстань до прямої між першою і останньою точками.
    """
    x = np.array(ks, dtype=np.float64)
    y = np.array(inertias, dtype=np.float64)

    x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)

    p1 = np.array([x_n[0], y_n[0]])
    p2 = np.array([x_n[-1], y_n[-1]])

    def point_line_dist(p):
        return np.abs(np.cross(p2 - p1, p1 - p)) / (np.linalg.norm(p2 - p1) + 1e-12)

    dists = [point_line_dist(np.array([x_n[i], y_n[i]])) for i in range(len(x_n))]
    return int(ks[int(np.argmax(dists))])


def choose_k_silhouette(ks: List[int], silhouettes: List[float]) -> int:
    """K за максимумом Silhouette Coefficient."""
    return int(ks[int(np.argmax(np.array(silhouettes)))])


# -----------------------------
# Візуалізація
# -----------------------------

def plot_curve(x: List[int], y: List[float], title: str, xlabel: str, ylabel: str, out_path: str) -> None:
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------
# Оцінка для списку K (inertia, silhouette, метрики)
# -----------------------------

def evaluate_for_k_list(
    orig_img: np.ndarray,
    work_img: np.ndarray,
    ks: List[int],
    seed: int,
    sample_pixels: Optional[int],
    out_dir: str,
    prefix: str,
    ref_bytes_for_cr: int,
    save_each_k_png: bool = True,
) -> Tuple[List[Dict], List[float], List[float]]:
    """
    orig_img: оригінал (без шуму) — для PSNR/SSIM з оригіналом (за потреби)
    work_img: те, що стискаємо (або чисте, або зашумлене)
    ref_bytes_for_cr: з чим порівнюємо CR за байтами (orig файл або orig_png тощо)

    Повертає:
      results: список рядків для CSV (включно з CR фактичним)
      inertias, silhouettes: для графіків і вибору K
    """
    h, w, _ = work_img.shape

    # Підвибірка для silhouette
    pixels = work_img.reshape(-1, 3).astype(np.float32)
    n = pixels.shape[0]
    sil_sample = min(10000, n)
    rng = np.random.default_rng(seed)
    sil_idx = rng.choice(n, size=sil_sample, replace=False)
    sil_pixels = pixels[sil_idx]

    results = []
    inertias = []
    silhouettes = []

    for k in ks:
        comp, inertia = kmeans_compress(work_img, k=k, seed=seed, sample_pixels=sample_pixels)

        # Якість відносно "бази":
        # - якщо work_img == orig_img (clean): метрики якості = між orig і comp
        # - якщо work_img noisy: метрики якості можна рахувати по-різному.
        #   Для диплома логічно оцінювати: (orig_img vs comp_noisy_k) — наскільки відновили "структуру".
        mse, psnr = mse_psnr(orig_img, comp)
        ssim_val = ssim_rgb(orig_img, comp)

        cr_theor = compression_ratio_theoretical(h, w, k)

        # silhouette на підвибірці
        model = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300, algorithm="lloyd")
        model.fit(sil_pixels)
        sil = float(silhouette_score(sil_pixels, model.labels_))

        # Збереження стисненого зображення і фактичний CR (за байтами)
        out_png_path = os.path.join(out_dir, f"{prefix}_k{k}.png")
        if save_each_k_png:
            save_rgb(comp, out_png_path)
        comp_bytes = file_size_bytes(out_png_path) if save_each_k_png else 0

        cr_actual = (ref_bytes_for_cr / comp_bytes) if (save_each_k_png and comp_bytes > 0) else float("nan")

        results.append({
            "k": int(k),
            "cr_theoretical": cr_theor,
            "cr_actual_bytes": cr_actual,
            "ref_bytes": int(ref_bytes_for_cr),
            "compressed_png_bytes": int(comp_bytes),
            "mse_vs_original": float(mse),
            "psnr_db_vs_original": float(psnr),
            "ssim_vs_original": float(ssim_val),
            "inertia": float(inertia),
            "silhouette": float(sil),
        })
        inertias.append(float(inertia))
        silhouettes.append(float(sil))

    return results, inertias, silhouettes


# -----------------------------
# Головна обробка одного зображення
# -----------------------------

def process_one_image(
    img_path: str,
    out_root: str,
    ks: List[int],
    sigmas: List[float],
    seed: int,
    sample_pixels: Optional[int],
    global_rows: List[Dict],
) -> None:
    name = os.path.splitext(os.path.basename(img_path))[0]
    img = load_rgb(img_path)
    h, w, _ = img.shape

    # Папки
    base_dir = os.path.join(out_root, name)
    clean_dir = os.path.join(base_dir, "clean")
    noise_root = os.path.join(base_dir, "noise")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noise_root, exist_ok=True)

    # --- Еталонні байти для CR ---
    # 1) Розмір оригінального файлу (як є)
    orig_bytes = file_size_bytes(img_path)

    # 2) Додатково збережемо "оригінал як PNG" для чеснішого порівняння (PNG vs PNG)
    orig_png_path = os.path.join(clean_dir, f"{name}_original_as_png.png")
    save_rgb(img, orig_png_path)
    orig_png_bytes = file_size_bytes(orig_png_path)

    # ---------------- CLEAN ----------------
    # Будемо рахувати CR фактичний двома способами:
    # - CR_actual_vs_origfile = orig_bytes / comp_png_bytes
    # - CR_actual_vs_orignpng = orig_png_bytes / comp_png_bytes  (рекомендовано для однакового формату)
    #
    # Для зручності зробимо два набори результатів, але в глобальну таблицю запишемо ОБИДВА.

    # 1) Відносно orig file bytes
    results_clean, inertias, silhouettes = evaluate_for_k_list(
        orig_img=img,
        work_img=img,
        ks=ks,
        seed=seed,
        sample_pixels=sample_pixels,
        out_dir=clean_dir,
        prefix=f"{name}_clean",
        ref_bytes_for_cr=orig_bytes,
        save_each_k_png=True,
    )
    k_elbow = choose_k_elbow(ks, inertias)
    k_sil = choose_k_silhouette(ks, silhouettes)

    # Порахувати CR_actual_vs_orignpng (png vs png): для цього просто перераховуємо з уже збережених файлів
    for r in results_clean:
        k = r["k"]
        comp_path = os.path.join(clean_dir, f"{name}_clean_k{k}.png")
        comp_bytes = file_size_bytes(comp_path)
        r["cr_actual_vs_orig_png"] = (orig_png_bytes / comp_bytes) if comp_bytes > 0 else float("nan")
        r["orig_png_bytes"] = orig_png_bytes
        r["orig_file_bytes"] = orig_bytes
        r["mode"] = "clean"
        r["sigma"] = 0
        r["image"] = name
        r["width"] = w
        r["height"] = h
        r["k_elbow"] = k_elbow
        r["k_silhouette_best"] = k_sil

        global_rows.append(r)

    # CSV clean
    csv_clean = os.path.join(clean_dir, f"{name}_clean_metrics.csv")
    with open(csv_clean, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image","width","height","mode","sigma",
            "k","k_elbow","k_silhouette_best",
            "cr_theoretical",
            "cr_actual_bytes","cr_actual_vs_orig_png",
            "orig_file_bytes","orig_png_bytes","compressed_png_bytes",
            "mse_vs_original","psnr_db_vs_original","ssim_vs_original",
            "inertia","silhouette",
        ]
        wri = csv.DictWriter(f, fieldnames=fieldnames)
        wri.writeheader()
        for r in results_clean:
            # доповнимо рядок потрібними полями, якщо чогось немає
            row = {k2: r.get(k2, "") for k2 in fieldnames}
            wri.writerow(row)

    # Графіки clean
    plot_curve(ks, inertias,
               title=f"{name}: Elbow method (Inertia vs K) [clean]",
               xlabel="K", ylabel="Inertia",
               out_path=os.path.join(clean_dir, f"{name}_elbow_clean.png"))
    plot_curve(ks, silhouettes,
               title=f"{name}: Silhouette score vs K [clean]",
               xlabel="K", ylabel="Silhouette",
               out_path=os.path.join(clean_dir, f"{name}_silhouette_clean.png"))

    # Summary clean
    with open(os.path.join(clean_dir, f"{name}_clean_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== ПІДСУМОК (CLEAN) ===\n")
        f.write(f"Файл: {img_path}\n")
        f.write(f"Розмір: {w}x{h}\n")
        f.write(f"orig_file_bytes: {orig_bytes}\n")
        f.write(f"orig_png_bytes:  {orig_png_bytes}\n")
        f.write(f"K: {ks}\n")
        f.write(f"Оптимальне K (Elbow): {k_elbow}\n")
        f.write(f"Оптимальне K (Silhouette): {k_sil}\n")
        f.write("Примітка: CR_actual_vs_orig_png (PNG vs PNG) — рекомендоване порівняння.\n")

    # ---------------- NOISE ----------------
    for sigma in sigmas:
        sigma_i = int(sigma)
        sigma_dir = os.path.join(noise_root, f"sigma_{sigma_i}")
        os.makedirs(sigma_dir, exist_ok=True)

        noisy = add_awgn(img, sigma=float(sigma), seed=seed)
        noisy_path = os.path.join(sigma_dir, f"{name}_noisy_sigma{sigma_i}.png")
        save_rgb(noisy, noisy_path)

        # Для фактичного CR за байтами:
        # зашумлений "вхід" можна рахувати двома способами:
        # 1) порівняння відносно ОРИГІНАЛУ (часто так і роблять, щоб показати кінцеву якість/стиснення відносно базового)
        # 2) порівняння відносно NOISY PNG (скільки стискаємо саме зашумлене)
        #
        # Зробимо обидва. Спочатку ref = orig_png_bytes (база), і додатково перерахуємо для noisy_bytes.

        noisy_bytes = file_size_bytes(noisy_path)

        results_noisy, inertias_n, silhouettes_n = evaluate_for_k_list(
            orig_img=img,          # оцінка якості ВІДНОСНО ОРИГІНАЛУ (як у дослідженнях)
            work_img=noisy,        # стискаємо зашумлене
            ks=ks,
            seed=seed,
            sample_pixels=sample_pixels,
            out_dir=sigma_dir,
            prefix=f"{name}_noisy_sigma{sigma_i}",
            ref_bytes_for_cr=orig_png_bytes,  # базовий CR: відносно orig_png
            save_each_k_png=True,
        )

        k_elbow_n = choose_k_elbow(ks, inertias_n)
        k_sil_n = choose_k_silhouette(ks, silhouettes_n)

        # доповнення полів + додатковий CR відносно noisy png
        for r in results_noisy:
            k = r["k"]
            comp_path = os.path.join(sigma_dir, f"{name}_noisy_sigma{sigma_i}_k{k}.png")
            comp_bytes = file_size_bytes(comp_path)
            r["cr_actual_vs_orig_png"] = (orig_png_bytes / comp_bytes) if comp_bytes > 0 else float("nan")
            r["cr_actual_vs_noisy_png"] = (noisy_bytes / comp_bytes) if comp_bytes > 0 else float("nan")

            r["orig_png_bytes"] = orig_png_bytes
            r["noisy_png_bytes"] = noisy_bytes
            r["orig_file_bytes"] = orig_bytes
            r["mode"] = "noise"
            r["sigma"] = sigma_i
            r["image"] = name
            r["width"] = w
            r["height"] = h
            r["k_elbow"] = k_elbow_n
            r["k_silhouette_best"] = k_sil_n

            global_rows.append(r)

        # CSV noisy
        csv_noisy = os.path.join(sigma_dir, f"{name}_noisy_sigma{sigma_i}_metrics.csv")
        with open(csv_noisy, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "image","width","height","mode","sigma",
                "k","k_elbow","k_silhouette_best",
                "cr_theoretical",
                "cr_actual_bytes","cr_actual_vs_orig_png","cr_actual_vs_noisy_png",
                "orig_file_bytes","orig_png_bytes","noisy_png_bytes","compressed_png_bytes",
                "mse_vs_original","psnr_db_vs_original","ssim_vs_original",
                "inertia","silhouette",
            ]
            wri = csv.DictWriter(f, fieldnames=fieldnames)
            wri.writeheader()
            for r in results_noisy:
                row = {k2: r.get(k2, "") for k2 in fieldnames}
                wri.writerow(row)

        # Графіки noisy
        plot_curve(ks, inertias_n,
                   title=f"{name}: Elbow (Inertia vs K) [sigma={sigma_i}]",
                   xlabel="K", ylabel="Inertia",
                   out_path=os.path.join(sigma_dir, f"{name}_elbow_sigma{sigma_i}.png"))
        plot_curve(ks, silhouettes_n,
                   title=f"{name}: Silhouette vs K [sigma={sigma_i}]",
                   xlabel="K", ylabel="Silhouette",
                   out_path=os.path.join(sigma_dir, f"{name}_silhouette_sigma{sigma_i}.png"))

        # Summary noisy
        with open(os.path.join(sigma_dir, f"{name}_noisy_sigma{sigma_i}_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"=== ПІДСУМОК (NOISE sigma={sigma_i}) ===\n")
            f.write(f"Базовий файл: {img_path}\n")
            f.write(f"Розмір: {w}x{h}\n")
            f.write(f"orig_png_bytes:  {orig_png_bytes}\n")
            f.write(f"noisy_png_bytes: {noisy_bytes}\n")
            f.write(f"K: {ks}\n")
            f.write(f"Оптимальне K (Elbow): {k_elbow_n}\n")
            f.write(f"Оптимальне K (Silhouette): {k_sil_n}\n")
            f.write("PSNR/SSIM пораховані відносно ОРИГІНАЛУ (clean), що дозволяє оцінити вплив шуму.\n")


# -----------------------------
# Збір зображень і зведена таблиця
# -----------------------------

def list_images(input_dir: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = []
    for fn in os.listdir(input_dir):
        p = os.path.join(input_dir, fn)
        if os.path.isfile(p) and os.path.splitext(fn.lower())[1] in exts:
            paths.append(p)
    paths.sort()
    return paths


def write_global_csv(global_rows: List[Dict], out_path: str) -> None:
    """Один зведений CSV по всіх зображеннях/σ/K."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = [
        "image","width","height","mode","sigma",
        "k","k_elbow","k_silhouette_best",
        "cr_theoretical",
        "cr_actual_vs_orig_png","cr_actual_vs_noisy_png",
        "orig_file_bytes","orig_png_bytes","noisy_png_bytes","compressed_png_bytes",
        "mse_vs_original","psnr_db_vs_original","ssim_vs_original",
        "inertia","silhouette",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=fieldnames)
        wri.writeheader()
        for r in global_rows:
            row = {k2: r.get(k2, "") for k2 in fieldnames}
            wri.writerow(row)


def plot_global_summary(global_rows: List[Dict], out_dir: str) -> None:

    os.makedirs(out_dir, exist_ok=True)

    # Групуємо за (mode,sigma,k) середні PSNR/SSIM
    # mode: clean/noise
    from collections import defaultdict

    psnr_map = defaultdict(list)
    ssim_map = defaultdict(list)

    for r in global_rows:
        mode = r.get("mode", "")
        sigma = int(r.get("sigma", 0) or 0)
        k = int(r.get("k", 0) or 0)
        psnr = float(r.get("psnr_db_vs_original", np.nan))
        ssimv = float(r.get("ssim_vs_original", np.nan))

        key = (mode, sigma, k)
        if np.isfinite(psnr):
            psnr_map[key].append(psnr)
        if np.isfinite(ssimv):
            ssim_map[key].append(ssimv)

    # Витягнемо всі k
    ks = sorted({int(r.get("k")) for r in global_rows if r.get("k") is not None})

    # Для clean
    for mode, sigma in sorted({(r.get("mode",""), int(r.get("sigma",0) or 0)) for r in global_rows}):
        # зберемо серію по k
        psnr_series = []
        ssim_series = []
        for k in ks:
            key = (mode, sigma, k)
            psnr_series.append(float(np.mean(psnr_map[key])) if psnr_map[key] else np.nan)
            ssim_series.append(float(np.mean(ssim_map[key])) if ssim_map[key] else np.nan)

        tag = "clean" if mode == "clean" else f"noise_sigma{sigma}"
        plot_curve(ks, psnr_series,
                   title=f"Середній PSNR vs K [{tag}]",
                   xlabel="K", ylabel="PSNR (dB)",
                   out_path=os.path.join(out_dir, f"avg_psnr_vs_k_{tag}.png"))
        plot_curve(ks, ssim_series,
                   title=f"Середній SSIM vs K [{tag}]",
                   xlabel="K", ylabel="SSIM",
                   out_path=os.path.join(out_dir, f"avg_ssim_vs_k_{tag}.png"))


# -----------------------------
# main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Дослідження стиснення K-means (CR, PSNR, SSIM, Elbow, Silhouette, AWGN)."
    )
    parser.add_argument("--input-dir", required=True, help="Папка з тестовими зображеннями (Lena, House, Mandrill, Airplane, Peppers тощо).")
    parser.add_argument("--output-dir", default="out", help="Папка результатів.")
    parser.add_argument("--ks", default="4,8,16,32,64", help="Список K через кому.")
    parser.add_argument("--sigmas", default="5,10,15,20,30", help="Sigma шуму через кому (СКВ).")
    parser.add_argument("--seed", type=int, default=42, help="Seed для відтворюваності.")
    parser.add_argument("--sample-pixels", type=int, default=50000,
                        help="Пікселів для навчання KMeans (для швидкості). 0 -> всі пікселі.")
    parser.add_argument("--make-global-plots", action="store_true",
                        help="Створити зведені графіки середніх PSNR/SSIM vs K (по всіх зображеннях).")

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Папка не знайдена: {args.input_dir}")

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    sigmas = [float(x.strip()) for x in args.sigmas.split(",") if x.strip()]

    sample_pixels = args.sample_pixels
    if sample_pixels <= 0:
        sample_pixels = None

    images = list_images(args.input_dir)
    if not images:
        raise RuntimeError("У input-dir не знайдено зображень (png/jpg/bmp/tiff).")

    print("Знайдені зображення:")
    for p in images:
        print(" -", p)

    global_rows: List[Dict] = []

    for img_path in images:
        print(f"\nОбробка: {img_path}")
        process_one_image(
            img_path=img_path,
            out_root=args.output_dir,
            ks=ks,
            sigmas=sigmas,
            seed=args.seed,
            sample_pixels=sample_pixels,
            global_rows=global_rows,
        )

    # Зведений CSV по всьому експерименту
    global_csv = os.path.join(args.output_dir, "GLOBAL_RESULTS.csv")
    write_global_csv(global_rows, global_csv)
    print("\nЗведений CSV:", global_csv)

    # Зведені графіки (опційно)
    if args.make_global_plots:
        global_plot_dir = os.path.join(args.output_dir, "GLOBAL_PLOTS")
        plot_global_summary(global_rows, global_plot_dir)
        print("Зведені графіки:", global_plot_dir)

    print("\nГотово. Результати у:", args.output_dir)


if __name__ == "__main__":
    main()