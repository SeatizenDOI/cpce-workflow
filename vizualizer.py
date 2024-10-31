import tkinter as tk
from tkinter import messagebox
from pathlib import Path
import matplotlib
import csv
import enum
import pandas as pd

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageDraw

# Global vars
WINDOW = [1600, 1200, 0, 0]  # [width, height, x_pos, y_pos]
FIG_SIZE = [6, 3]  # inches
DATA_PATH = Path("./csv/2_clean_salary.csv")
SAVE_PATH = Path('./csv/anno.csv')  # path to file with .csv annotations (will be created)

AUTOSAVE_TIME = 60  # seconds

class Status(enum.Enum):
    UNRESOLVED = 0
    ACCEPTED = 1
    REJECTED = 2


def parse_cpce_file(filepath):
    
    rows = []
    with open(filepath, "r", encoding="latin1") as file:
        rows = [a.strip() for a in file.readlines()]
    
    min_, max_ = [float(b) for b in rows[1].split(",")]

    nb_point = int(rows[5])
    points_value = []
    for i in range (6 , nb_point + 6):
        x, y = [float(a) for a in rows[i].split(',')]
        value = rows[i + nb_point].split(',')[1].replace('"', '')
        points_value.append({"x": x, "y": y, "v": value})

    return points_value, min_, max_

class DataItem:
    """ Items with the same filename are considered the same, even though their parent directory is different """

    def __init__(self, fname, status=Status.UNRESOLVED):
        self.fname = fname
        self.status = status

    def __repr__(self):
        return self.fname.name

    def __hash__(self):
        return hash(self.fname.name)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.fname.name == other.fname.name

    def accept(self):
        self.status = Status.ACCEPTED

    def reject(self):
        self.status = Status.REJECTED

    def unresolve(self):
        self.status = Status.UNRESOLVED

# tk classes
class Images:
    def __init__(self, data_path):
        self.fig = plt.Figure(figsize=FIG_SIZE, dpi=300, tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.data_path = data_path
        self.annot_path = SAVE_PATH
        self.load_data() # Load self.data

        self.index = 0
        self.img_size = (0, 0)
        self.labels = []
        self.needToDraw = True

    def __len__(self):
        return len(self.data)

    def update_image(self):
        """ Show one image """
        self.ax.cla()
        fname_actual = self.get_file_path_actual()
        img = Image.open(fname_actual)
    
        draw = ImageDraw.Draw(img)
        colours = {"S": (255,0,0), "ALGAE": (125,255,0), "R": (0, 128, 255), "SG": (128, 0, 128)}
        points_value, min_, max_ = parse_cpce_file(self.get_file_cpce_actual())
        self.labels = list({a["v"] for a in points_value})
        self.img_size = img.size
        w, h = img.size
        ratio = max_ / h
        for points in points_value:
            center_x, center_y = points["x"] / ratio, points["y"] / ratio
            half_diag = h / 10
            draw.circle((center_x, center_y), h / 100, fill=colours.get(points["v"], (255,255,255)))
            if not self.needToDraw: continue
            if points["x"] >= max_ or points["x"] <= min_ or points["y"] >= max_ or points["y"] <= min_: continue
            draw.rectangle([center_x - half_diag, center_y - half_diag, center_x + half_diag, center_y + half_diag], fill=None, outline=(0,0,255))

        self.ax.imshow(img)

    def get_file_path_actual(self):
        return self.data.iloc[self.index]["relative_file_path"]

    def get_file_cpce_actual(self):
        return self.data.iloc[self.index]["cpce_file"]

    def reject_image(self):
        self.data.loc[self.index, "status"] = Status.REJECTED

    def accept_image(self):
        self.data.loc[self.index, "status"] = Status.ACCEPTED

    def unresolve_image(self):
        self.data.loc[self.index, "status"] = Status.UNRESOLVED

    def get_filename_actual(self):
        return self.data.iloc[self.index]["FileName"]

    def get_status_actual(self):
        return self.data.iloc[self.index]["status"]
    
    def toggle_need_to_draw(self):
        self.needToDraw = not(self.needToDraw)

    def get_labels_of_the_frame(self):
        return ", ".join(self.labels)
    
    def getStatusFromDataFrame(self, x, status):
        try:
            row = status[status["filename"] == x["FileName"]]
            if len(row) == 0: return Status.UNRESOLVED
            
            if row.iloc[0]["status"] == Status.UNRESOLVED.value: return Status.UNRESOLVED
            elif row.iloc[0]["status"] == Status.ACCEPTED.value: return Status.ACCEPTED
            elif row.iloc[0]["status"] == Status.REJECTED.value: return Status.REJECTED

        except:
            return Status.UNRESOLVED

    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        status = pd.read_csv(self.annot_path)
        self.data["status"] = self.data.apply(lambda x: self.getStatusFromDataFrame(x, status), axis=1)

    def save_data(self):
        fields = ['filename', 'status']
        with open(self.annot_path, 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows([[item["FileName"], item["status"].value] for i, item in self.data.iterrows()])

class Text:
    def __init__(self):
        self.text = ""

    def update_text(self, val):
        self.text = val

    def get_text(self):
        return self.text
    
class Control:
    def __init__(self):
        self.annot = 1
        self.seen = 1

class MySlideShow(tk.Tk):
    def __init__(self, *args, **kwargs):
        # tk params
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry('{}x{}+{}+{}'.format(WINDOW[0], WINDOW[1], WINDOW[2], WINDOW[3]))
        self.title("Aina vizualizer")

        # data structures
        self.images = Images(DATA_PATH)
        self.text_fname = Text()
        self.text_label = Text()
        self.text_status = Text()
        

        # tk windows
        self.image_window_actual = FigureCanvasTkAgg(self.images.fig, self)
        self.text_fname_window = tk.Label(self, text=self.text_fname.get_text())
        self.text_label_window = tk.Label(self, text=self.text_label.get_text())
        self.text_status_window = tk.Label(self, text=self.text_status.get_text())
        self.previous_button = tk.Button(self, text="Previous (←)", command=self.previous_index)
        self.next_button = tk.Button(self, text="Next (→)", command=self.next_index)
        self.seen_button = tk.Button(self, text="Toggle Seen (S)", command=self.toggle_seen)
        self.reject_button = tk.Button(self, text="Reject (X)", command=self.reject)
        self.accept_button = tk.Button(self, text="Accept (Space)", command=self.accept)
        self.clear_button = tk.Button(self, text="Clear Flag (U)", command=self.unresolve)

        # positioning
        self.image_window_actual.get_tk_widget().grid(row=0, column=0, sticky=tk.N, columnspan=4)
        self.text_fname_window.grid(row=1, column=0, sticky=tk.N, columnspan=4)
        self.text_label_window.grid(row=2, column=0, sticky=tk.N, columnspan=4)
        self.text_status_window.grid(row=2, column=1, sticky=tk.N, columnspan=4)
        self.previous_button.grid(row=3, column=0, sticky=tk.N, pady=(5, 0))
        self.next_button.grid(row=3, column=1, sticky=tk.N, pady=(5, 0))
        self.seen_button.grid(row=3, column=3, sticky=tk.N, pady=(10, 0))
        self.reject_button.grid(row=4, column=0, sticky=tk.N, pady=(5, 0))
        self.accept_button.grid(row=4, column=1, sticky=tk.N, pady=(5, 0))
        self.clear_button.grid(row=4, column=2, sticky=tk.N, pady=(5, 0))

        # key bindings
        # self.focus_set()
        self.bind('<Right>', self.next_index_keypress)
        self.bind('<Left>', self.previous_index_keypress)
        self.bind('x', self.reject_keypress)
        self.bind('X', self.reject_keypress)
        self.bind('<space>', self.accept_keypress)
        self.bind('u', self.unresolve_keypress)
        self.bind('U', self.unresolve_keypress)

        self.autosave()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # init
        self.update_frame()

    def reject(self):
        self.images.reject_image()
        self.next_index()

    def reject_keypress(self, event):
        self.reject()

    def accept(self):
        self.images.accept_image()
        self.next_index()

    def accept_keypress(self, event):
        self.accept()

    def unresolve(self):
        self.images.unresolve_image()
        self.update_frame()

    def unresolve_keypress(self, event):
        self.unresolve()
    
    def toggle_seen(self):
        self.images.toggle_need_to_draw()
        self.update_frame()

    def toggle_seen_keypress(self, event):
        self.toggle_seen()
    
    def next_index(self):
        self.images.index = (self.images.index + 1) % len(self.images.data)
        self.update_frame()

    def next_index_keypress(self, event):
        self.next_index()

    def previous_index(self):
        self.images.index = (self.images.index - 1) % len(self.images.data)
        self.update_frame()

    def previous_index_keypress(self, event):
        self.previous_index()

    def update_frame(self):
        self.images.update_image()

        self.text_fname.update_text(f"{self.images.get_file_path_actual()}, {self.images.index}/{len(self.images.data)}")
        self.text_fname_window.configure(text=self.text_fname.get_text())

        self.text_status.update_text(self.images.get_status_actual())
        self.text_status_window.configure(text=self.text_status.get_text())

        self.text_label.update_text(self.images.get_labels_of_the_frame())
        self.text_label_window.configure(text=self.text_label.get_text())

        self.image_window_actual.draw()
    
    def done(self):
        print("It's all done :)")
        exit(0)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to Save & Quit?"):
            self.images.save_data()
            self.destroy()

    def autosave(self):
        self.images.save_data()
        self.after(AUTOSAVE_TIME * 1000, self.autosave)

    def run(self):
        self.mainloop()


if __name__ == '__main__':
    if not DATA_PATH.exists():
        print(DATA_PATH, " does not exist.")
        exit(1)

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    app = MySlideShow()
    app.run()
