import os
import sys
import queue
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PhotoDuplicateFinder.photo_dedup import process_photos


class PhotoDuplicateGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('相似照片整理工具')
        self.root.geometry('620x450')
        self.root.resizable(False, False)

        self.folder_var = tk.StringVar()
        self.threshold_var = tk.IntVar(value=8)
        self.status_var = tk.StringVar(value='準備就緒')
        self.percent_var = tk.StringVar(value='0%')
        self.events = queue.Queue()
        self.running = False
        self.completed_folder = None

        self.build_ui()
        self.root.after(100, self.check_events)

    def build_ui(self):
        main = ttk.Frame(self.root, padding=25)
        main.pack(fill='both', expand=True)

        ttk.Label(main, text='相似照片整理工具', font=('Arial', 18, 'bold')).pack(
            anchor='w', pady=(0, 20)
        )

        ttk.Label(main, text='選擇照片資料夾：').pack(anchor='w')
        folder_frame = ttk.Frame(main)
        folder_frame.pack(fill='x', pady=(6, 18))

        self.folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var)
        self.folder_entry.pack(side='left', fill='x', expand=True)
        self.browse_button = ttk.Button(
            folder_frame, text='瀏覽…', command=self.browse_folder
        )
        self.browse_button.pack(side='left', padx=(8, 0))

        ttk.Label(main, text='pHash 距離門檻：').pack(anchor='w')
        threshold_frame = ttk.Frame(main)
        threshold_frame.pack(fill='x', pady=(6, 4))

        self.threshold_scale = ttk.Scale(
            threshold_frame, from_=0, to=20,
            orient='horizontal', command=self.update_threshold
        )
        self.threshold_scale.pack(side='left', fill='x', expand=True)
        # 必須先建立標籤再設定滑桿值，避免初始化時觸發 callback 發生錯誤。
        self.threshold_label = ttk.Label(threshold_frame, text='8', width=4)
        self.threshold_label.pack(side='left', padx=10)
        self.threshold_scale.set(self.threshold_var.get())

        ttk.Label(
            main, text='數值越小，判定越嚴格；數值越大，越容易被判為相似。',
            foreground='gray'
        ).pack(anchor='w', pady=(0, 22))

        self.start_button = ttk.Button(
            main, text='開始整理照片', command=self.start_processing
        )
        self.start_button.pack(fill='x', ipady=8)

        progress_header = ttk.Frame(main)
        progress_header.pack(fill='x', pady=(20, 5))
        ttk.Label(progress_header, text='處理進度').pack(side='left')
        ttk.Label(
            progress_header, textvariable=self.percent_var,
            font=('Arial', 11, 'bold')
        ).pack(side='right')

        self.progress = ttk.Progressbar(
            main, orient='horizontal', mode='determinate', maximum=100, value=0
        )
        self.progress.pack(fill='x', pady=(0, 10))
        ttk.Label(main, textvariable=self.status_var, wraplength=560).pack(anchor='w')

        self.open_button = ttk.Button(
            main, text='開啟輸出資料夾', command=self.open_output,
            state='disabled'
        )
        self.open_button.pack(anchor='e', pady=(25, 0))

    def update_threshold(self, value):
        threshold = int(round(float(value)))
        self.threshold_var.set(threshold)
        self.threshold_label.config(text=str(threshold))

    def browse_folder(self):
        folder = filedialog.askdirectory(title='選擇照片資料夾')
        if folder:
            self.folder_var.set(folder)
            self.completed_folder = None
            self.open_button.config(state='disabled')

    def update_progress(self, percent, message):
        # 由背景執行緒呼叫；只透過 Queue 傳送，避免直接操作 Tkinter。
        self.events.put(('progress', percent, self.translate_status(message)))

    @staticmethod
    def translate_status(message):
        """將 photo_dedup.py 回報的常見英文進度轉成繁體中文。"""
        translations = {
            'Scanning image directory...': '正在掃描照片資料夾…',
            'Running agglomerative clustering...': '正在進行階層式分群…',
            'Clustering complete': '分群完成',
            'Writing photo_groups.csv...': '正在寫入照片分組紀錄…',
            'Processing completed successfully!': '照片整理完成！',
            'Reading images:': '正在讀取照片：',
            'Calculating distances:': '正在計算照片距離：',
            'Copying photos:': '正在複製照片：',
        }
        result = str(message)
        for english, chinese in translations.items():
            result = result.replace(english, chinese)
        result = result.replace(' pairs', ' 組配對')
        return result

    def set_running(self, running):
        self.running = running
        state = 'disabled' if running else 'normal'
        for widget in (
            self.start_button, self.browse_button,
            self.folder_entry, self.threshold_scale
        ):
            widget.config(state=state)

    def start_processing(self):
        if self.running:
            return
        folder = self.folder_var.get().strip()
        if not folder or not Path(folder).is_dir():
            messagebox.showerror('資料夾無效', '請先選擇有效的照片資料夾。')
            return

        self.set_running(True)
        self.completed_folder = None
        self.open_button.config(state='disabled')
        self.progress['value'] = 0
        self.percent_var.set('0%')
        self.status_var.set('正在啟動照片分析…')

        worker = threading.Thread(
            target=self.process_worker,
            args=(folder, self.threshold_var.get()), daemon=True
        )
        worker.start()

    def process_worker(self, folder, threshold):
        try:
            stats = process_photos(
                directory=folder,
                threshold=threshold,
                progress_callback=self.update_progress,
            )
            self.events.put(('success', stats))
        except Exception as exc:
            self.events.put(('error', str(exc)))

    def check_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == 'progress':
                    _, percent, message = event
                    self.progress['value'] = percent
                    self.percent_var.set(f'{percent:.1f}%')
                    self.status_var.set(message)
                elif kind == 'success':
                    stats = event[1]
                    self.set_running(False)
                    self.completed_folder = stats['output_directory']
                    self.open_button.config(state='normal')
                    self.progress['value'] = 100
                    self.percent_var.set('100%')
                    self.status_var.set(
                        f"整理完成！不重複照片：{stats['unique']} 張；"
                        f"相似照片：{stats['duplicate']} 張。"
                    )
                    messagebox.showinfo(
                        '整理完成',
                        f"照片總數：{stats['total']} 張\n"
                        f"不重複照片：{stats['unique']} 張\n"
                        f"相似照片：{stats['duplicate']} 張\n"
                        f"相似照片群組：{stats['duplicate_groups']} 組\n"
                        f"略過的照片：{stats['skipped']} 張\n\n"
                        f"輸出位置：{self.completed_folder}"
                    )
                elif kind == 'error':
                    self.set_running(False)
                    self.status_var.set('處理失敗，請確認錯誤訊息。')
                    messagebox.showerror('處理失敗', event[1])
        except queue.Empty:
            pass
        self.root.after(100, self.check_events)

    def open_output(self):
        folder = self.completed_folder
        if not folder or not Path(folder).is_dir():
            return
        if sys.platform == 'win32':
            os.startfile(folder)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', folder])
        else:
            subprocess.Popen(['xdg-open', folder])


if __name__ == '__main__':
    root = tk.Tk()
    app = PhotoDuplicateGUI(root)
    root.mainloop()
