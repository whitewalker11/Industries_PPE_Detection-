import os
import cv2
import json
import logging
import tkinter as tk
#from tkinter import Tk, ttk, filedialog, Label, StringVar, BooleanVar, messagebox
from threading import Thread
from pathlib import Path
from PIL import ImageTk, Image
import nsvision as nv
from tkinter import *
#import tkMessageBox
#import Tkinter



import tkinter
import tkinter.ttk
from time import strftime

def ui_ppe():
    root = tkinter.Tk()

    OPTIONS = [
        "ProductionHouse1",
        "Productionhouse2",
        "Productionhouse3"
    ]

    def var_states():
        # print("helmet: %d\nmask: %d\ngloves:%d" % (self.checkBoxVar1.get(), self.checkBoxVar2.get(),self.checkBoxVar3.get()))

        root.destroy()
        root.panel.quit()
        root.quit()

    root.wm_iconbitmap('solar.ico')
    root.title("CheckBox for PPEs model")
    root.geometry("800x500")  # set starting size of window
    root.config(bg="skyblue")

    log = Label(root, text="PPEs Detection", bg="#2176C1", fg='white', relief=RAISED)
    log.pack(ipady=5, fill='x')
    log.config(font=("Font", 30))

    root.panel = tkinter.ttk.Frame(root)
    root.panel.pack(fill=tkinter.BOTH, expand=1)

    T = tk.Text(root.panel, height=1, width=18)
    T.pack()
    T.place(x=30,y=10+20)
    T.insert(tk.END, "Select PPEs Model")

    T = tk.Text(root.panel, height=1, width=24)
    T.pack()
    T.place(x=30+100+350+50,y=10+20)
    T.insert(tk.END, "Select Production House")


    root.checkBoxVar1 = tkinter.BooleanVar()
    root.checkBoxVar1.set(False)

    root.checkBox1 = tkinter.ttk.Checkbutton(root.panel, text="Helmet", variable=root.checkBoxVar1)
    root.checkBox1.place(x=30, y=40+30)

    root.checkBoxVar2 = tkinter.BooleanVar()
    root.checkBoxVar2.set(False)

    root.checkBox2 = tkinter.ttk.Checkbutton(root.panel, text="Mask", variable=root.checkBoxVar2)
    root.checkBox2.place(x=30, y=70+30)

    root.checkBoxVar3 = tkinter.BooleanVar()
    root.checkBoxVar3.set(False)

    root.checkBox3 = tkinter.ttk.Checkbutton(root.panel, text="Gloves", variable=root.checkBoxVar3)
    root.checkBox3.place(x=30, y=100+30)

    root.checkBoxVar4 = tkinter.BooleanVar()
    root.checkBoxVar4.set(False)

    root.checkBox4 = tkinter.ttk.Checkbutton(root.panel, text="Boots", variable=root.checkBoxVar4)
    root.checkBox4.place(x=30, y=130+30)

    '''root.checkBoxVar5=tkinter.BooleanVar()
    root.checkBoxVar5.set(False)
    root.checkBox5 = tkinter.ttk.Checkbutton(root.panel, text="Production House 1", variable=root.checkBoxVar5)
    root.checkBox5.place(x=530,y=70)

    root.checkBoxVar6=tkinter.BooleanVar()
    root.checkBoxVar6.set(False)
    root.checkBox6= tkinter.ttk.Checkbutton(root.panel, text="Production House 2", variable=root.checkBoxVar6)
    root.checkBox6.place(x=530,y=70+30)

    root.checkBoxVar7 = tkinter.BooleanVar()
    root.checkBoxVar7.set(False)
    root.checkBox7 = tkinter.ttk.Checkbutton(root.panel, text="Production House 3", variable=root.checkBoxVar7)
    root.checkBox7.place(x=530, y=70 + 30+30)'''

    # self.lineRed = tkinter.LabelFrame(self.panel, borderwidth=1, background="red", height=600, width=2)
    # self.lineRed.place(x=10, y=10)
    tkinter.ttk.Button(root.panel,width=15,text='Proceed', command=var_states).pack(side=BOTTOM)





    variable = StringVar(root.panel)
    variable.set(OPTIONS[0])  # default value

    w = OptionMenu(root.panel,variable, *OPTIONS)
    w.pack(side=TOP)
    w.place(x=30+100+400,y=40-5+30)

    def ok():
        print("value is:" + variable.get())

    button = Button(root.panel, text="OK", command=ok)
    button.pack(side=TOP)
    button.place(x=100+180+400,y=43-5+30)



    lbl = Label(root,background = 'lightblue')

    # Placing clock at the centre
    # of the tkinter window
    lbl.pack()


    # self.geometry("300x300+200+100")

    # def main(self=None):

    root.mainloop()
    l = []
    st=[]
    l.append(bool(root.checkBoxVar1.get()))
    l.append(bool(root.checkBoxVar2.get()))
    l.append(bool(root.checkBoxVar3.get()))
    l.append(bool(root.checkBoxVar4.get()))

    st=variable.get()

    #print(l)

    return l,st


