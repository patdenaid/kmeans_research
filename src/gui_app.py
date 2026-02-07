# gui_app.py
# Графічний застосунок (Tkinter) для запуску дослідження K-means стиснення.
# Запуск:
#   python gui_app.py
#
# У тій же папці (src/) мають бути:
#   - diploma_kmeans_research.py
#   - controller.py

from __future__ import annotations
import os
import sys
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from controller import ResearchController, RunConfig, parse_int_list, parse_float_list


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("K-means Image Compression Research (Диплом)")
        self.geometry("980x640")
        self.minsize(900, 600)

        self.controller = ResearchController()
        self.ui_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()

        self._build_ui()
        self._poll_ui_queue()

    # ---------------- UI ----------------

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        # Верхня панель налаштувань
        cfg_frame = ttk.LabelFrame(root, text="Налаштування")
        cfg_frame.pack(fill="x", **pad)

        # input-dir
        self.input_var = tk.StringVar(value=os.path.abspath("images"))
        ttk.Label(cfg_frame, text="Папка зображень (input-dir):").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(cfg_frame, textvariable=self.input_var, width=70).grid(row=0, column=1, sticky="we", padx=8, pady=6)
        ttk.Button(cfg_frame, text="Обрати…", command=self._choose_input).grid(row=0, column=2, padx=8, pady=6)

        # output-dir
        self.output_var = tk.StringVar(value=os.path.abspath("out"))
        ttk.Label(cfg_frame, text="Папка результатів (output-dir):").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(cfg_frame, textvariable=self.output_var, width=70).grid(row=1, column=1, sticky="we", padx=8, pady=6)
        ttk.Button(cfg_frame, text="Обрати…", command=self._choose_output).grid(row=1, column=2, padx=8, pady=6)

        # K list
        self.ks_var = tk.StringVar(value="4,8,16,32,64")
        ttk.Label(cfg_frame, text="K (через кому):").grid(row=2, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(cfg_frame, textvariable=self.ks_var, width=30).grid(row=2, column=1, sticky="w", padx=8, pady=6)

        # Sigma list
        self.sigmas_var = tk.StringVar(value="5,10,15,20,30")
        ttk.Label(cfg_frame, text="Sigma шуму (через кому):").grid(row=3, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(cfg_frame, textvariable=self.sigmas_var, width=30).grid(row=3, column=1, sticky="w", padx=8, pady=6)

        # seed, sample_pixels
        self.seed_var = tk.StringVar(value="42")
        self.sample_var = tk.StringVar(value="50000")

        row4 = ttk.Frame(cfg_frame)
        row4.grid(row=4, column=0, columnspan=3, sticky="we", padx=8, pady=6)

        ttk.Label(row4, text="Seed:").pack(side="left")
        ttk.Entry(row4, textvariable=self.seed_var, width=8).pack(side="left", padx=(6, 18))

        ttk.Label(row4, text="Пікселів для навчання KMeans (0 = всі):").pack(side="left")
        ttk.Entry(row4, textvariable=self.sample_var, width=10).pack(side="left", padx=6)

        # global plots
        self.global_plots_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cfg_frame, text="Створити зведені графіки (GLOBAL_PLOTS)", variable=self.global_plots_var)\
            .grid(row=5, column=0, columnspan=3, sticky="w", padx=8, pady=6)

        cfg_frame.columnconfigure(1, weight=1)

        # Кнопки керування
        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill="x", **pad)

        self.run_btn = ttk.Button(btn_frame, text="▶ Запустити дослідження", command=self._on_run)
        self.run_btn.pack(side="left", padx=6)

        self.stop_btn = ttk.Button(btn_frame, text="⏹ Зупинити", command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)

        self.open_out_btn = ttk.Button(btn_frame, text="📂 Відкрити папку результатів", command=self._open_output_dir)
        self.open_out_btn.pack(side="right", padx=6)

        # Прогрес
        prog_frame = ttk.Frame(root)
        prog_frame.pack(fill="x", **pad)
        ttk.Label(prog_frame, text="Статус:").pack(side="left")

        self.status_var = tk.StringVar(value="Готово до запуску.")
        ttk.Label(prog_frame, textvariable=self.status_var).pack(side="left", padx=8)

        self.pbar = ttk.Progressbar(prog_frame, mode="indeterminate")
        self.pbar.pack(side="right", fill="x", expand=True, padx=8)

        # Лог
        log_frame = ttk.LabelFrame(root, text="Лог виконання")
        log_frame.pack(fill="both", expand=True, **pad)

        self.log_text = tk.Text(log_frame, wrap="word", height=18)
        self.log_text.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scroll.set)

        self._log("Готово. Обери папку зображень та натисни 'Запустити дослідження'.")

    # ---------------- Helpers ----------------

    def _choose_input(self):
        p = filedialog.askdirectory(title="Обрати папку зображень (input-dir)")
        if p:
            self.input_var.set(p)

    def _choose_output(self):
        p = filedialog.askdirectory(title="Обрати папку результатів (output-dir)")
        if p:
            self.output_var.set(p)

    def _open_output_dir(self):
        path = self.output_var.get().strip()
        if not path:
            return
        os.makedirs(path, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f'open "{path}"')
            else:
                os.system(f'xdg-open "{path}"')
        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалося відкрити папку:\n{e}")

    def _log(self, msg: str):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    # ---------------- Thread-safe UI updates ----------------

    def _enqueue_log(self, msg: str):
        self.ui_queue.put(("log", msg))

    def _enqueue_done(self, ok: bool, msg: str):
        self.ui_queue.put(("done", f"{int(ok)}|{msg}"))

    def _poll_ui_queue(self):
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "log":
                    self._log(payload)
                    self.status_var.set(payload[:120] + ("…" if len(payload) > 120 else ""))
                elif kind == "done":
                    ok_s, msg = payload.split("|", 1)
                    ok = bool(int(ok_s))
                    self._on_done(ok, msg)
        except queue.Empty:
            pass
        self.after(100, self._poll_ui_queue)

    # ---------------- Actions ----------------

    def _on_run(self):
        if self.controller.is_running():
            return

        try:
            input_dir = self.input_var.get().strip()
            output_dir = self.output_var.get().strip()
            ks = parse_int_list(self.ks_var.get())
            sigmas = parse_float_list(self.sigmas_var.get())
            seed = int(self.seed_var.get().strip() or "42")

            sample_raw = self.sample_var.get().strip()
            sample_pixels = int(sample_raw) if sample_raw else 50000
            if sample_pixels <= 0:
                sample_pixels = None

            cfg = RunConfig(
                input_dir=input_dir,
                output_dir=output_dir,
                ks=ks,
                sigmas=sigmas,
                seed=seed,
                sample_pixels=sample_pixels,
                make_global_plots=bool(self.global_plots_var.get()),
            )

        except Exception as e:
            messagebox.showerror("Помилка параметрів", str(e))
            return

        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.pbar.start(10)
        self.status_var.set("Запуск...")

        self._log("\n==============================")
        self._log("▶ Запуск дослідження")
        self._log(f"input-dir: {cfg.input_dir}")
        self._log(f"output-dir: {cfg.output_dir}")
        self._log(f"K: {cfg.ks}")
        self._log(f"sigma: {cfg.sigmas}")
        self._log(f"seed: {cfg.seed}, sample_pixels: {cfg.sample_pixels}")
        self._log("==============================\n")

        self.controller.run_async(
            cfg=cfg,
            log=self._enqueue_log,
            done=lambda ok, msg: self._enqueue_done(ok, msg),
        )

    def _on_stop(self):
        if self.controller.is_running():
            self.controller.request_stop()
            self._log("⏹ Запит на зупинку надіслано...")

    def _on_done(self, ok: bool, msg: str):
        self.pbar.stop()
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set(msg)
        if ok:
            self._log(f"\n✅ {msg}\n")
            messagebox.showinfo("Готово", msg)
        else:
            self._log(f"\n❌ {msg}\n")
            messagebox.showwarning("Завершено", msg)


if __name__ == "__main__":
    app = App()
    app.mainloop()