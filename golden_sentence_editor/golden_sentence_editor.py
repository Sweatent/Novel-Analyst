import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
try:
    import sv_ttk
except ImportError:
    messagebox.showerror("缺少依赖", "请先安装sv-ttk库: pip install sv-ttk")
    exit()

class GoldenSentenceEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("金句审核工具")
        self.geometry("900x650")

        # Set a modern font
        self.style = ttk.Style()
        self.style.configure("TLabel", font=("Microsoft YaHei UI", 10))
        self.style.configure("TButton", font=("Microsoft YaHei UI", 10))
        self.style.configure("TEntry", font=("Microsoft YaHei UI", 10))
        self.style.configure("TCheckbutton", font=("Microsoft YaHei UI", 10))
        self.style.configure("TLabelframe.Label", font=("Microsoft YaHei UI", 11, "bold"))

        self.data = []
        self.current_index = -1
        self.progress_file = ""
        self.output_file = ""

        self._create_menu()
        self._create_widgets()
        self.set_dark_theme() # Set default theme after widgets are created
        self._update_ui()

    def set_light_theme(self):
        sv_ttk.set_theme("light")

    def set_dark_theme(self):
        sv_ttk.set_theme("dark")

    def _create_menu(self):
        menu_bar = tk.Menu(self)
        self.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开金句JSON文件...", command=self.load_golden_sentences)
        file_menu.add_command(label="加载进度...", command=self.load_progress)
        file_menu.add_separator()
        file_menu.add_command(label="保存进度", command=self.save_progress)
        file_menu.add_command(label="另存为最终结果...", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.on_closing)
        
        view_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="视图", menu=view_menu)
        view_menu.add_command(label="浅色模式", command=self.set_light_theme)
        view_menu.add_command(label="深色模式", command=self.set_dark_theme)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Speaker
        speaker_frame = ttk.Frame(main_frame)
        speaker_frame.pack(fill=tk.X, pady=5)
        ttk.Label(speaker_frame, text="说话人:", width=10).pack(side=tk.LEFT)
        self.speaker_var = tk.StringVar()
        self.speaker_entry = ttk.Entry(speaker_frame, textvariable=self.speaker_var)
        self.speaker_entry.pack(fill=tk.X, expand=True)

        # Sentence
        ttk.Label(main_frame, text="句子:").pack(fill=tk.X, pady=(10, 0))
        self.sentence_text = tk.Text(main_frame, wrap=tk.WORD, height=15, font=("Microsoft YaHei UI", 11), relief=tk.FLAT)
        self.sentence_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Buttons
        button_frame = ttk.Frame(main_frame, padding="10 0")
        button_frame.pack(fill=tk.X)

        self.prev_button = ttk.Button(button_frame, text="上一条", command=self.prev_sentence)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        self.next_button = ttk.Button(button_frame, text="下一条", command=self.next_sentence)
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.approve_button = ttk.Button(button_frame, text="通过", command=self.approve_sentence)
        self.approve_button.pack(side=tk.RIGHT, padx=5)
        self.reject_button = ttk.Button(button_frame, text="不通过", command=self.reject_sentence)
        self.reject_button.pack(side=tk.RIGHT, padx=5)
        
        # Advanced actions
        action_frame = ttk.Frame(main_frame, padding="10 0")
        action_frame.pack(fill=tk.X, pady=10)
        self.context_button = ttk.Button(action_frame, text="查询上下文", command=self.show_context)
        self.context_button.pack(side=tk.LEFT, padx=5)
        self.merge_button = ttk.Button(action_frame, text="合并...", command=self.open_merge_window)
        self.merge_button.pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def on_closing(self):
        self._commit_changes()
        if messagebox.askyesno("退出", "您想在退出前保存进度吗？"):
            self.save_progress()
        self.destroy()

    def _update_ui(self):
        if not self.data or self.current_index < 0:
            self.speaker_var.set("")
            self.sentence_text.delete("1.0", tk.END)
            self.status_var.set("请先打开金句文件")
            for child in self.winfo_children():
                if isinstance(child, ttk.Frame):
                    for widget in child.winfo_children():
                        if isinstance(widget, (ttk.Button, ttk.Entry, tk.Text)):
                            widget.config(state=tk.DISABLED)
            return

        for child in self.winfo_children():
            if isinstance(child, ttk.Frame):
                for widget in child.winfo_children():
                    if isinstance(widget, (ttk.Button, ttk.Entry, tk.Text)):
                        widget.config(state=tk.NORMAL)

        item = self.data[self.current_index]
        self.speaker_var.set(item.get("speaker", ""))
        self.sentence_text.delete("1.0", tk.END)
        self.sentence_text.insert("1.0", item.get("sentence", ""))
        
        chapter_title = item.get('chapter_title', '未知章节')
        status = f"章节: {chapter_title} | 进度: {self.current_index + 1} / {len(self.data)}"
        if 'status' in item:
            status += f" | 状态: {item['status']}"
        self.status_var.set(status)

        self.prev_button.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_index < len(self.data) - 1 else tk.DISABLED)

    def load_golden_sentences(self):
        filepath = filedialog.askopenfilename(
            title="打开金句JSON文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            if not isinstance(raw_data, dict) or "chapters" not in raw_data:
                messagebox.showerror("错误", "JSON文件格式不正确，需要包含 'chapters' 键。")
                self.data = []
                return

            self.data = []
            for chapter in raw_data.get("chapters", []):
                chapter_title = chapter.get("chapter_title", "未知章节")
                chapter_context = chapter.get("original_title", "")
                all_golden_in_chapter = chapter.get("golden_sentences", [])

                for sentence_data in all_golden_in_chapter:
                    sentence_data['context'] = chapter_context
                    sentence_data['chapter_title'] = chapter_title
                    sentence_data['siblings'] = all_golden_in_chapter
                    self.data.append(sentence_data)

            self.current_index = 0
            self.progress_file = f"{os.path.splitext(filepath)[0]}_progress.json"
            self.output_file = f"{os.path.splitext(filepath)[0]}_result.json"
            self.title(f"金句审核工具 - {os.path.basename(filepath)}")
            self._update_ui()
            messagebox.showinfo("成功", f"成功加载 {len(self.data)} 条金句。")

        except (json.JSONDecodeError, FileNotFoundError) as e:
            messagebox.showerror("错误", f"加载文件失败: {e}")
            self.data = []
        
        self._update_ui()

    def load_progress(self):
        filepath = filedialog.askopenfilename(
            title="加载进度文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            # Reconstruct the data with siblings
            loaded_data = progress_data['data']
            self.current_index = progress_data['current_index']
            
            # Group sentences by chapter
            chapters = {}
            for item in loaded_data:
                chapter_title = item.get('chapter_title', '未知章节')
                if chapter_title not in chapters:
                    chapters[chapter_title] = []
                chapters[chapter_title].append(item)
            
            # Rebuild the main data list and add siblings
            self.data = []
            for chapter_title, sentences_in_chapter in chapters.items():
                for sentence_data in sentences_in_chapter:
                    # The 'siblings' are all sentences in the same chapter
                    sentence_data['siblings'] = sentences_in_chapter
                    self.data.append(sentence_data)

            # Ensure the order is preserved if possible (optional but good practice)
            # This simple re-append might change order if chapters were mixed in original file.
            # For this editor's purpose, it should be fine.

            self.progress_file = filepath
            self.output_file = f"{os.path.splitext(filepath)[0]}_result.json"
            self.title(f"金句审核工具 - {os.path.basename(filepath)}")
            self._update_ui()
            messagebox.showinfo("成功", "进度加载成功。")

        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            messagebox.showerror("错误", f"加载进度失败: {e}")
        
        self._update_ui()

    def save_progress(self):
        self._commit_changes()
        if not self.data:
            messagebox.showwarning("警告", "没有数据可保存。")
            return

        # Create a deep copy for saving, removing circular references
        data_to_save = []
        for item in self.data:
            clean_item = item.copy()
            clean_item.pop('siblings', None)
            data_to_save.append(clean_item)

        progress_data = {
            'current_index': self.current_index,
            'data': data_to_save
        }
        
        # If progress file is not set, ask user where to save it
        if not self.progress_file:
            filepath = filedialog.asksaveasfilename(
                title="保存进度",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not filepath:
                return # User cancelled
            self.progress_file = filepath
            self.output_file = f"{os.path.splitext(filepath)[0]}_result.json"
            self.title(f"金句审核工具 - {os.path.basename(filepath)}")
        
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=4)
            messagebox.showinfo("成功", f"进度已保存到 {self.progress_file}")
        except Exception as e:
            messagebox.showerror("错误", f"保存进度失败: {e}")

    def save_results(self):
        self._commit_changes()
        filepath = filedialog.asksaveasfilename(
            title="另存为最终结果",
            initialfile=self.output_file,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return

        approved_sentences_raw = [s for s in self.data if s.get('status') == 'approved']
        
        approved_sentences_clean = []
        for s in approved_sentences_raw:
            clean_s = s.copy()
            clean_s.pop('siblings', None)
            approved_sentences_clean.append(clean_s)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(approved_sentences_clean, f, ensure_ascii=False, indent=4)
            messagebox.showinfo("成功", f"成功保存 {len(approved_sentences_clean)} 条已通过的金句。")
        except Exception as e:
            messagebox.showerror("错误", f"保存文件失败: {e}")

    def _commit_changes(self):
        if self.current_index >= 0 and self.data:
            item = self.data[self.current_index]
            item['speaker'] = self.speaker_var.get()
            item['sentence'] = self.sentence_text.get("1.0", tk.END).strip()

    def prev_sentence(self):
        self._commit_changes()
        if self.current_index > 0:
            self.current_index -= 1
            self._update_ui()

    def next_sentence(self):
        self._commit_changes()
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            self._update_ui()
        else:
            # Reached the end of the list
            if messagebox.askyesno("完成", "所有金句已审核完毕！\n是否现在保存最终结果？"):
                self.save_results()

    def approve_sentence(self):
        self._commit_changes()
        if self.current_index >= 0 and self.data:
            self.data[self.current_index]['status'] = 'approved'
            self.next_sentence()

    def reject_sentence(self):
        self._commit_changes()
        if self.current_index >= 0 and self.data:
            self.data[self.current_index]['status'] = 'rejected'
            self.next_sentence()
        
    def show_context(self):
        if self.current_index < 0 or not self.data:
            return

        item = self.data[self.current_index]
        context = item.get("context")

        if not context:
            messagebox.showinfo("无上下文", "当前条目没有可用的上下文信息。")
            return

        context_win = tk.Toplevel(self)
        context_win.title("上下文")
        context_win.geometry("600x400")
        
        text_widget = tk.Text(context_win, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        if isinstance(context, list):
            context_text = "\n".join(context)
        else:
            context_text = str(context)
            
        text_widget.insert(tk.END, context_text)
        
        sentence = item.get("sentence", "")
        start_pos = "1.0"
        while True:
            start_pos = text_widget.search(sentence, start_pos, stopindex=tk.END)
            if not start_pos:
                break
            end_pos = f"{start_pos}+{len(sentence)}c"
            text_widget.tag_add("highlight", start_pos, end_pos)
            start_pos = end_pos
            
        text_widget.tag_config("highlight", background="yellow", foreground="black")
        text_widget.config(state=tk.DISABLED)

    def open_merge_window(self):
        if self.current_index < 0 or not self.data:
            return
        
        current_item = self.data[self.current_index]
        MergeWindow(self, current_item)

class MergeWindow(tk.Toplevel):
    def __init__(self, parent, current_item):
        super().__init__(parent)
        self.parent = parent
        self.current_item = current_item
        
        self.title("合并句子")
        self.geometry("1000x700")

        self.grab_set()
        self._create_widgets()
        self._populate_data()

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        left_frame = ttk.LabelFrame(main_frame, text="金句列表 (勾选以合并)")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        self.golden_list_frame = ttk.Frame(left_frame)
        self.golden_list_frame.grid(row=0, column=0, sticky="nsew")

        right_frame = ttk.LabelFrame(main_frame, text="文章上下文")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        self.context_text = tk.Text(right_frame, wrap=tk.WORD, padx=5, pady=5, font=("Microsoft YaHei UI", 10), relief=tk.FLAT)
        self.context_text.grid(row=0, column=0, sticky="nsew")
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        ttk.Button(button_frame, text="执行合并", command=self.perform_merge).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="取消", command=self.destroy).pack(side=tk.RIGHT)

    def _populate_data(self):
        context = self.current_item.get("context", "无可用上下文。")
        self.context_text.insert("1.0", context)
        self.context_text.config(state=tk.DISABLED)

        self.check_vars = []
        all_golden = self.current_item.get("siblings", [])
        for i, sentence_data in enumerate(all_golden):
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(self.golden_list_frame, text=sentence_data.get("sentence", ""), variable=var)
            cb.pack(anchor="w", fill="x")
            self.check_vars.append((var, sentence_data))

    def perform_merge(self):
        selected_items = [item for var, item in self.check_vars if var.get()]

        if len(selected_items) < 2:
            messagebox.showwarning("警告", "请至少选择两个句子进行合并。")
            return

        merged_sentence = "".join(item.get("sentence", "") for item in selected_items)
        
        base_item = selected_items[0]
        base_item['sentence'] = merged_sentence
        
        items_to_remove = selected_items[1:]
        
        ids_to_remove = {id(item) for item in items_to_remove}
        self.parent.data = [item for item in self.parent.data if id(item) not in ids_to_remove]

        try:
            new_index = self.parent.data.index(base_item)
            self.parent.current_index = new_index
        except ValueError:
            self.parent.current_index = 0

        self.parent._update_ui()
        messagebox.showinfo("成功", "句子已成功合并。")
        self.destroy()

if __name__ == "__main__":
    app = GoldenSentenceEditor()
    app.mainloop()