
from __future__ import annotations
import os
import threading
import traceback
from dataclasses import dataclass
from typing import Callable, List, Optional, Dict

from diploma_kmeans_research import (
    list_images,
    process_one_image,
    write_global_csv,
    plot_global_summary,
)

LogFn = Callable[[str], None]
DoneFn = Callable[[bool, str], None]


@dataclass
class RunConfig:
    input_dir: str
    output_dir: str
    ks: List[int]
    sigmas: List[float]
    seed: int = 42
    sample_pixels: Optional[int] = 50000  # None -> всі пікселі
    make_global_plots: bool = False


def parse_int_list(s: str) -> List[int]:
    items = []
    for part in s.split(","):
        part = part.strip()
        if part:
            items.append(int(part))
    if not items:
        raise ValueError("Список K порожній.")
    return items


def parse_float_list(s: str) -> List[float]:
    items = []
    for part in s.split(","):
        part = part.strip()
        if part:
            items.append(float(part))
    if not items:
        raise ValueError("Список sigma порожній.")
    return items


def validate_config(cfg: RunConfig) -> None:
    if not os.path.isdir(cfg.input_dir):
        raise FileNotFoundError(f"Не знайдено папку зображень: {cfg.input_dir}")
    os.makedirs(cfg.output_dir, exist_ok=True)

    if any(k < 2 for k in cfg.ks):
        raise ValueError("Усі K мають бути >= 2.")

    if any(s < 0 for s in cfg.sigmas):
        raise ValueError("Sigma шуму має бути >= 0.")

    if cfg.sample_pixels is not None and cfg.sample_pixels <= 0:
        cfg.sample_pixels = None  # трактуємо як "всі пікселі"


class ResearchController:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def request_stop(self) -> None:
        self._stop_flag.set()

    def run_async(self, cfg: RunConfig, log: LogFn, done: DoneFn) -> None:
        if self.is_running():
            log("⚠️ Процес уже запущено.")
            return

        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._run_impl,
            args=(cfg, log, done),
            daemon=True
        )
        self._thread.start()

    def _run_impl(self, cfg: RunConfig, log: LogFn, done: DoneFn) -> None:
        try:
            validate_config(cfg)

            imgs = list_images(cfg.input_dir)
            if not imgs:
                raise RuntimeError("У папці input-dir не знайдено зображень (png/jpg/bmp/tiff).")

            log("✅ Знайдені зображення:")
            for p in imgs:
                log(f"   • {os.path.basename(p)}")

            global_rows: List[Dict] = []

            for i, img_path in enumerate(imgs, start=1):
                if self._stop_flag.is_set():
                    log("⏹️ Зупинено користувачем.")
                    done(False, "Зупинено")
                    return

                log(f"\n▶ Обробка ({i}/{len(imgs)}): {img_path}")
                process_one_image(
                    img_path=img_path,
                    out_root=cfg.output_dir,
                    ks=cfg.ks,
                    sigmas=cfg.sigmas,
                    seed=cfg.seed,
                    sample_pixels=cfg.sample_pixels,
                    global_rows=global_rows,
                )
                log(f"✔ Готово: {os.path.basename(img_path)}")

            global_csv = os.path.join(cfg.output_dir, "GLOBAL_RESULTS.csv")
            write_global_csv(global_rows, global_csv)
            log(f"\n📄 Збережено зведений CSV: {global_csv}")

            if cfg.make_global_plots:
                plot_dir = os.path.join(cfg.output_dir, "GLOBAL_PLOTS")
                plot_global_summary(global_rows, plot_dir)
                log(f"📊 Збережено зведені графіки: {plot_dir}")

            done(True, f"Готово! Результати у: {cfg.output_dir}")

        except Exception as e:
            log("❌ Помилка під час виконання:")
            log(str(e))
            log("\nТехнічні деталі:\n" + traceback.format_exc())
            done(False, "Помилка виконання")