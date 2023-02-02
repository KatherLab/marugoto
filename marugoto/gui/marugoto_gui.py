#!/usr/bin/env python

__author__ = "Gregory Veldhuizen"
__copyright__ = "Copyright 2022, Kather Lab"
__maintainer__ = "Gregory Veldhuizen"

from cgi import test
from cgitb import text
from doctest import script_from_examples
from re import T
from sqlite3 import Row
from string import whitespace
from tkinter import *
from tkinter.filedialog import askdirectory, askopenfile, askopenfilename
from tkinter.tix import COLUMN
import pandas as pd
import subprocess
import os

# this is to open the programme
root = Tk()
root.title("Marugoto")
root.geometry("1000x660")
root.eval("tk::PlaceWindow . center")
root.resizable(False, False)


# this is to enable the correct buttons depending on which tool is selected
def tool_selection(selection):
    selection = choice.get()
    if selection == "Attention MIL Training":
        clini_file_button.config(state="normal")
        slide_file_button.config(state="normal")
        feature_directory_button.config(state="normal")
        output_directory_button.config(state="normal")
        model_file_button.config(state=DISABLED)
        numberoffolds.config(state=DISABLED)
        stats_checkbox.config(state=DISABLED)
        statsvar.set(0)
        roc_checkbox.config(state=DISABLED)
        rocvar.set(0)

    elif selection == "Attention MIL Deployment":
        clini_file_button.config(state="normal")
        slide_file_button.config(state="normal")
        feature_directory_button.config(state="normal")
        model_file_button.config(state="normal")
        output_directory_button.config(state="normal")
        numberoffolds.config(state=DISABLED)
        stats_checkbox.config(state=NORMAL)
        statsvar.set(1)
        roc_checkbox.config(state=NORMAL)
        rocvar.set(1)

    elif selection == "Attention MIL Cross-Validation":
        clini_file_button.config(state="normal")
        slide_file_button.config(state="normal")
        feature_directory_button.config(state="normal")
        output_directory_button.config(state="normal")
        numberoffolds.config(state="normal")
        stats_checkbox.config(state=NORMAL)
        statsvar.set(1)
        roc_checkbox.config(state=NORMAL)
        rocvar.set(1)


# this is for selecting the tool
toolslist = [
    "Attention MIL Training",
    "Attention MIL Deployment",
    "Attention MIL Cross-Validation",
]
choice = StringVar(root, "")
choice.set("Choose Tool")
tooltype = OptionMenu(root, choice, *toolslist, command=tool_selection)
tooltype.config(width=25, height=3, font=100)
tooltype.grid(column=0, row=0)


def select_clini():
    global clini_file
    clini_file = askopenfile(
        filetypes=[("excel file", "*.xlsx")], initialdir="~/Documents"
    )
    clini_file_path_label1.set(clini_file.name)
    # below for getting target label names
    clini_column_names = list((pd.read_excel(str(clini_file.name)).columns.values))
    options = clini_column_names
    global chosen_target
    chosen_target = StringVar()
    chosen_target.set("Target Category")
    target_label = OptionMenu(root, chosen_target, *options)
    target_label.config(font=100, width=25, height=3)
    target_label.grid(column=0, row=4)
    dummy_target_label.destroy()
    dummy_target_label1.destroy()


def select_slide():
    slide_file = askopenfile(
        filetypes=[("csv file", "*.csv")], initialdir="~/Documents"
    )
    slide_file_path_label1.set(slide_file.name)


def select_feature():
    feature_folder = askdirectory(initialdir="~/Documents")
    feature_file_path_label1.set(str(feature_folder))


def select_model():
    model_file = askopenfile(
        filetypes=[("pkl file", "*.pkl")], initialdir="~/Documents"
    )
    model_file_path_label1.set(str(model_file.name))


def select_output():
    output_folder = askdirectory(initialdir="~/Documents")
    output_file_path_label1.set(str(output_folder))


def run_tool():
    source = os.path.dirname(__file__)
    filename = "temp_gui_script.sh"
    filepath = os.path.join(source, filename)
    initial_command = (
        "#! /bin/bash" + "\nsource /home/$USER/.virtualenvs/marugoto/bin/activate"
    )

    final_command = (
        '\nread -p "Task done! Press any key to exit..."'
        + "\nrm "
        + filepath
        + "\nexit"
    )

    # information needed for generating stats and roc curves
    clinidf = pd.read_excel(str(clini_file.name))
    targetcolumn = chosen_target.get()
    targetclasses = clinidf[targetcolumn].dropna().unique().tolist()

    # functions for generating stats and roc curves for deployment and xval tasks
    def generate_stats_script_xval():
        stats_script_xval = (
            "\npython3 -m marugoto.stats.categorical "
            + output_file_path_label1.get()
            + "/fold-*/patient-preds.csv "
            + "--outpath "
            + output_file_path_label1.get()
            + " --target-label "
            + chosen_target.get()
        )
        return stats_script_xval

    def generate_roc_script_xval():
        roc_script_xval = ""
        for targetclass in targetclasses:
            roc_script_xval += (
                "\npython3 -m marugoto.visualizations.roc "
                + output_file_path_label1.get()
                + "/fold-*/patient-preds.csv "
                + "--outpath "
                + output_file_path_label1.get()
                + " --target-label "
                + chosen_target.get()
                + " --true-label "
                + targetclass
            )
        return roc_script_xval

    def generate_stats_script_deployment():
        stats_script_deployment = (
            "\npython3 -m marugoto.stats.categorical "
            + output_file_path_label1.get()
            + "/patient-preds.csv "
            + "--outpath "
            + output_file_path_label1.get()
            + " --target-label "
            + chosen_target.get()
        )
        return stats_script_deployment

    def generate_roc_script_deployment():
        roc_script_deployment = ""
        for targetclass in targetclasses:
            roc_script_deployment += (
                "\npython3 -m marugoto.visualizations.roc "
                + output_file_path_label1.get()
                + "/patient-preds.csv "
                + "--outpath "
                + output_file_path_label1.get()
                + " --target-label "
                + chosen_target.get()
                + " --true-label "
                + targetclass
            )
        return roc_script_deployment

    if choice.get() == "Attention MIL Training":
        mil_training_command = (
            "\npython3 -m marugoto.mil train"
            + " --clini-table "
            + clini_file_path_label1.get()
            + " --slide-csv "
            + slide_file_path_label1.get()
            + " --feature-dir "
            + feature_file_path_label1.get()
            + " --target-label "
            + chosen_target.get()
            + " --output-path "
            + output_file_path_label1.get()
        )
        with open(filepath, "w") as text_file:
            text_file.write(initial_command + mil_training_command + final_command)

        runscriptcommand = "bash"
        root.destroy()
        subprocess.run([runscriptcommand, filepath])

    elif choice.get() == "Attention MIL Deployment":
        source = os.path.dirname(__file__)
        filename = "temp_gui_script.sh"
        filepath = os.path.join(source, filename)
        mil_training_command = (
            "\npython3 -m marugoto.mil deploy"
            + " --clini-table "
            + clini_file_path_label1.get()
            + " --slide-csv "
            + slide_file_path_label1.get()
            + " --feature-dir "
            + feature_file_path_label1.get()
            + " --target-label "
            + chosen_target.get()
            + " --model-path "
            + model_file_path_label1.get()
            + " --output-path "
            + output_file_path_label1.get()
        )

        if statsvar.get() == 1:
            mil_training_command += generate_stats_script_deployment()
        if rocvar.get() == 1:
            mil_training_command += generate_roc_script_deployment()
        with open(filepath, "w") as text_file:
            text_file.write(initial_command + mil_training_command + final_command)

        runscriptcommand = "bash"
        root.destroy()
        subprocess.run([runscriptcommand, filepath])

    elif choice.get() == "Attention MIL Cross-Validation":
        source = os.path.dirname(__file__)
        filename = "temp_gui_script.sh"
        filepath = os.path.join(source, filename)
        mil_training_command = (
            "\npython3 -m marugoto.mil crossval"
            + " --clini-excel "
            + clini_file_path_label1.get()
            + " --slide-csv "
            + slide_file_path_label1.get()
            + " --feature-dir "
            + feature_file_path_label1.get()
            + " --target-label "
            + chosen_target.get()
            + " --output-path "
            + output_file_path_label1.get()
            + " --n-splits "
            + (foldchoice.get())[0]
        )

        if statsvar.get() == 1:
            mil_training_command += generate_stats_script_xval()
        if rocvar.get() == 1:
            mil_training_command += generate_roc_script_xval()
        with open(filepath, "w") as text_file:
            text_file.write(initial_command + mil_training_command + final_command)

        with open(filepath, "w") as text_file:
            text_file.write(initial_command + mil_training_command + final_command)

        runscriptcommand = "bash"
        root.destroy()
        subprocess.run([runscriptcommand, filepath])
    else:
        print("HUH.")


clini_file_button = Button(
    root,
    text="Clini Table",
    command=select_clini,
    width=26,
    height=3,
    borderwidth=2,
    state="disabled",
)
clini_file_button.config(font=100)
clini_file_button.grid(column=0, rows=1)

slide_file_button = Button(
    root,
    text="Slide Table",
    command=select_slide,
    width=26,
    height=3,
    borderwidth=2,
    state="disabled",
)
slide_file_button.config(font=100)
slide_file_button.grid(column=0, row=2)

feature_directory_button = Button(
    root,
    text="Feature Directory",
    command=select_feature,
    width=26,
    height=3,
    borderwidth=2,
    state="disabled",
)
feature_directory_button.config(font=100)
feature_directory_button.grid(column=0, row=3)

foldnumbers = ["2 Folds", "3 Folds", "4 Folds", "5 Folds"]
foldchoice = StringVar(root, "")
foldchoice.set("Choose Number of Folds")
numberoffolds = OptionMenu(root, foldchoice, *foldnumbers, command=tool_selection)
numberoffolds.config(width=25, height=3, font=100, state="disabled")
numberoffolds.grid(column=0, row=6)

output_directory_button = Button(
    root,
    text="Output Directory",
    command=select_output,
    width=26,
    height=3,
    borderwidth=2,
    state="disabled",
)
output_directory_button.config(font=100)
output_directory_button.grid(column=0, row=7)

model_file_button = Button(
    root,
    text="Model File",
    command=select_model,
    width=26,
    height=3,
    borderwidth=2,
    state="disabled",
)
model_file_button.config(font=100)
model_file_button.grid(column=0, row=5)

runtool_button = Button(
    root,
    text="Run Marugoto",
    command=run_tool,
    width=26,
    height=3,
    borderwidth=2,
    state="normal",
)
runtool_button.config(font=100)
runtool_button.grid(column=0, row=10)

clini_file_path_label1 = StringVar(root, "")
clini_file_path_label2 = Label(root, textvariable=clini_file_path_label1).grid(
    column=1, row=1
)

slide_file_path_label1 = StringVar(root, "")
slide_file_path_label2 = Label(root, textvariable=slide_file_path_label1).grid(
    column=1, row=2
)


feature_file_path_label1 = StringVar(root, "")
feature_file_path_label2 = Label(root, textvariable=feature_file_path_label1).grid(
    column=1, row=3
)

dummy_targetcats = [""]
dummy_chosen_target = StringVar(root, "")
dummy_chosen_target.set("Target Category")
dummy_target_label = OptionMenu(root, dummy_chosen_target, *dummy_targetcats)
dummy_target_label.config(font=100, width=25, height=3, state="disabled")
dummy_target_label.grid(column=0, row=4)
dummy_target_label1 = Label(root, text="First Select Clini Table")
dummy_target_label1.grid(column=1, row=4)

model_file_path_label1 = StringVar(root, "")
model_file_path_label2 = Label(root, textvariable=model_file_path_label1).grid(
    column=1, row=5
)

output_file_path_label1 = StringVar(root, "")
output_file_path_label2 = Label(root, textvariable=output_file_path_label1).grid(
    column=1, row=7
)

# checkboxes for stats and graphs
checkboxframe = LabelFrame(root)
checkboxframe.grid(column=3, row=0, rowspan=1000)

statsvar = IntVar()
stats_checkbox = Checkbutton(
    checkboxframe,
    text="Calculate Statistics",
    variable=statsvar,
    onvalue=1,
    offvalue=0,
    state=DISABLED,
)
stats_checkbox.pack()

rocvar = IntVar()
roc_checkbox = Checkbutton(
    checkboxframe,
    text="Plot ROC Curve",
    variable=rocvar,
    onvalue=1,
    offvalue=0,
    state=DISABLED,
)
roc_checkbox.pack()


mainloop()
root.mainloop()
