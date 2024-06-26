import tkinter as tk
from tkinter import filedialog, messagebox
import configparser
from main import run_pipeline  # 确保导入 run_pipeline 函数


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Pipeline")

        self.config_file = "mixConfig.ini"
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file)

        self.steps = ["predicted", "expand", "extract", "stitch", "all"]
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Select Step:").grid(row=0, column=0, padx=10, pady=10)

        self.step_var = tk.StringVar(value=self.steps[0])
        self.step_menu = tk.OptionMenu(self.root, self.step_var, *self.steps)
        self.step_menu.grid(row=0, column=1, padx=10, pady=10)

        self.run_button = tk.Button(self.root, text="Run", command=self.run_step)
        self.run_button.grid(row=0, column=2, padx=10, pady=10)

        self.paths_frame = tk.LabelFrame(self.root, text="Paths")
        self.paths_frame.grid(
            row=1, column=0, columnspan=3, padx=10, pady=10, sticky="ew"
        )

        self.path_vars = {}
        self.path_entries = {}
        self.browse_buttons = {}

        for section in self.config.sections():
            for key in self.config[section]:
                var = tk.StringVar(value=self.config[section][key])
                self.path_vars[key] = var
                row = len(self.path_vars)
                tk.Label(self.paths_frame, text=key).grid(
                    row=row, column=0, padx=10, pady=5, sticky="e"
                )
                entry = tk.Entry(self.paths_frame, textvariable=var, width=50)
                entry.grid(row=row, column=1, padx=10, pady=5)
                self.path_entries[key] = entry
                button = tk.Button(
                    self.paths_frame,
                    text="Browse",
                    command=lambda k=key: self.browse_path(k),
                )
                button.grid(row=row, column=2, padx=10, pady=5)
                self.browse_buttons[key] = button

        self.save_button = tk.Button(
            self.root, text="Save Config", command=self.save_config
        )
        self.save_button.grid(row=2, column=1, padx=10, pady=10)

    def browse_path(self, key):
        path = (
            filedialog.askdirectory()
            if "dir" in key.lower()
            else filedialog.askopenfilename()
        )
        if path:
            self.path_vars[key].set(path)

    def save_config(self):
        for section in self.config.sections():
            for key in self.config[section]:
                self.config[section][key] = self.path_vars[key].get()
        with open(self.config_file, "w") as configfile:
            self.config.write(configfile)
        messagebox.showinfo("Info", "Config file saved!")

    def run_step(self):
        step = self.step_var.get()
        run_pipeline(step)
        messagebox.showinfo("Info", f"{step} step completed!")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
