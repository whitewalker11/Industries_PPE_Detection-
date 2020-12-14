from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter as tk
import os
import json
import csv
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

import tkinter as tk
from tkinter import ttk
file = ''
INPUT = ''
try:
    with open(os.path.join(ROOT_DIR,"settings.json"), 'r') as config_buffer:
        config = json.loads(config_buffer.read())
except FileNotFoundError:
    print("setting not found")

def GUI_final():
    def Take_input():
        global INPUT
        # INPUT = inputtxt.get("1.0", "end-1c")
        # print(INPUT)

        root.destroy()

    def open_file():
        global file
        file = filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          filetypes=(("csv files",
                                                      "*.csv*"),
                                                     ("all files",
                                                      "*.*")))
        if file is not None:
            lm = Label(text="file opend:" + file, bg="white", fg="black", font=("Arial", 9, "bold"), width=68, height=3)
            lm.place(x=100, y=200)
            # content = file.read()
            print(file)

    root = Tk()
    master = root.frame()
    background = '#303030'
    background2 = '#305035'

    root.geometry('1000x600')
    root.title("PPE DETECTION")
    root.configure(bg=background, bd=0)

    logo_image = ImageTk.PhotoImage(Image.open(config["gui"]["logo"]))
    il = Label(root, bg=background, image=logo_image)
    tl = Label(root, text="PPE KIT DETECTION", bg=background,
               fg="skyblue", font=("Arial", 24, "bold"))
    il.place(x=250 - 90+90, y=10 + 30)
    tl.place(x=400 - 140+90, y=29 + 30)

    l_csv = Label(text="Select Model to detect", bg="black", fg="white", font=("Times New Roman", 12, "bold"))
    l_csv.place(x=130+500, y=160+50)
    upload_image = ImageTk.PhotoImage(Image.open(config["gui"]["upload"]))
    ui = Label(root, bg=background, image=upload_image)
    ui.place(x=90+500, y=150+50)

    var1 = IntVar()
    b1 = Checkbutton(root, text="MASK", bg=background, font=("Times New Roman", 13, "bold"), onvalue=1, offvalue=0,
                     fg="grey", variable=var1)
    b1.place(x=100+600, y=200+50)

    var2 = IntVar()
    b2 = Checkbutton(root, text="HELMET", bg=background, font=("Times New Roman", 13, "bold"), onvalue=1, offvalue=0,
                     fg="grey", variable=var2)
    b2.place(x=100+600, y=240+50)

    var3 = IntVar()
    b3 = Checkbutton(root, text="GLOVES", bg=background, font=("Times New Roman", 13, "bold"), onvalue=1, offvalue=0,
                     fg="grey", variable=var3)
    b3.place(x=100+600, y=280+50)

    # btn = Button(root, text="Browse file", bg=background,
    # fg="white", font=("Arial", 8, "bold"), command=lambda: open_file())

    '''l = Label(text="Write your message for email", bg=background2, fg="white", font=("Arial", 8, "bold"))

    inputtxt = Text(root, height=15,
                    width=60,
                    bg="light yellow")'''

    # label
    drop_box = Label(root, text="Select the Production House:",
                     font=("Times New Roman", 13), bg="black", fg="white")
    drop_box.place(x=130, y=160+50)
    upload_image2 = ImageTk.PhotoImage(Image.open(config["gui"]["upload"]))
    ui2 = Label(root, bg=background, image=upload_image2)
    ui2.place(x=90, y=150+50)

    n = tk.StringVar()
    monthchoosen = ttk.Combobox(root, width=27, textvariable=n)

    # Adding combobox drop down list
    monthchoosen['values'] = (' PP12',
                              ' PP13',
                              )

    monthchoosen.current(0)
    monthchoosen.place(x=110, y=200+50)


    drop_box2 = Label(root, text="Select the camera Unit:",
                     font=("Times New Roman", 13), bg="black", fg="white")
    drop_box2.place(x=130, y=160+50+100)
    upload_image4 = ImageTk.PhotoImage(Image.open(config["gui"]["upload"]))
    ui4 = Label(root, bg=background, image=upload_image4)
    ui4.place(x=90, y=150+50+100)

    n1 = tk.StringVar()
    monthchoosen1 = ttk.Combobox(root, width=27, textvariable=n1)

    # Adding combobox drop down list
    monthchoosen1['values'] = (' Unit1',
                              ' Unit2',
                              )

    monthchoosen1.current(0)
    monthchoosen1.place(x=110, y=200+50+100)


    # Combobox creation

    upload_image3 = ImageTk.PhotoImage(Image.open('widgets/begin.png'))
    ui3 = Label(root, bg=background, image=upload_image3)
    ui3.place(x=460-30, y=500)
    Display = Button(root, height=2,
                     width=18,
                     text="PROCEED",
                     command=lambda: Take_input())

    # l.place(x=100, y=548 - 200)
    # btn.place(x=95, y=187)
    # inputtxt.place(x=100, y=570 - 200)
    Display.place(x=465-30, y=505)
    # Output.pack()

    mainloop()
    # print(INPUT)

    model=[]
    pro=n.get()
    cam=n1.get()
    model.append(var1)
    model.append(var2)
    model.append(var3)
    print(var1.get(),var2.get(),var3.get(),pro,cam)

    return model,cam,pro




