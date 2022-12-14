"""argparse with GUI"""

import tkinter as tk
from collections import OrderedDict
from tkinter import filedialog, messagebox
from typing import List


class ArgBase(tk.Frame):
    """"""
    @property
    def tip(self):
        raise NotImplementedError

    @tip.setter
    def tip(self, text):
        raise NotImplementedError

    @property
    def value(self):
        raise NotImplementedError

    @value.setter
    def value(self, value):
        raise NotImplementedError


class ValueArg(ArgBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.label = tk.Label(self)
        self.entry = tk.Entry(self)

        self.label.pack(side=tk.LEFT)
        self.entry.pack(side=tk.LEFT, expand=tk.YES, fill=tk.X)

    @property
    def tip(self):
        return self.label["text"][:-2]

    @tip.setter
    def tip(self, text):
        self.label["text"] = f"{text}: "

    @property
    def value(self):
        return self.entry.get()

    @value.setter
    def value(self, value):
        self.entry.delete(0, tk.END)
        self.entry.insert(0, value)


class FileArg(ValueArg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.btn = tk.Button(self, text="选择...", command=self.choose)
        self.btn.pack(side=tk.LEFT)

    def choose(self):
        self.value = filedialog.askopenfilename(initialdir=".")


class DirectoryArg(FileArg):
    def choose(self):
        self.value = filedialog.askdirectory(initialdir=".")


class BooleanArg(ArgBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.label = tk.Label(self)
        self.chkbtnvar = tk.BooleanVar(self)
        self.chkbtn = tk.Checkbutton(self, variable=self.chkbtnvar)

        self.label.pack(side=tk.LEFT)
        self.chkbtn.pack(side=tk.LEFT)

    @property
    def tip(self):
        return self.label["text"][:-2]

    @tip.setter
    def tip(self, text):
        self.label["text"] = f"{text}: "

    @property
    def value(self):
        return self.chkbtnvar.get()

    @value.setter
    def value(self, value):
        self.chkbtn.select() if value else self.chkbtn.deselect()


class ArgumentParserGUI:
    """"""

    def __init__(self) -> None:
        self._args = []

        self.positinal_args = OrderedDict()
        self.optional_args = OrderedDict()
        self.boolean_args = OrderedDict()

        self.root = tk.Tk()
        self.root.title("输入参数")

        self.value_arg_panel = tk.Frame(self.root)
        self.boolean_arg_panel = tk.Frame(self.root)
        self.confirm_btn = tk.Button(self.root, text="确认", command=self.confirm)

    def _init(self):
        """"""
        for w in self.positinal_args.values():
            w.pack(side=tk.TOP, fill=tk.X)

        for w in self.optional_args.values():
            w.pack(side=tk.TOP, fill=tk.X)

        for i, w in enumerate(self.boolean_args.values()):
            w.grid(row=(i//2), column=(i % 2))

        self.value_arg_panel.pack(side=tk.TOP, fill=tk.X)
        self.boolean_arg_panel.pack(side=tk.TOP, fill=tk.X)
        self.confirm_btn.pack(side=tk.TOP, fill=tk.X)

    def add_value_arg(self, name, tip="", default="", required=True):
        arg = ValueArg(self.value_arg_panel)
        arg.tip = tip or name.strip("-")
        arg.value = "" if required else default

        if required:
            self.positinal_args[name] = arg
        else:
            self.optional_args[name] = arg

    def add_file_arg(self, name, tip="", default="", required=True):
        arg = FileArg(self.value_arg_panel)
        arg.tip = tip or name.strip("-")
        arg.value = "" if required else default

        if required:
            self.positinal_args[name] = arg
        else:
            self.optional_args[name] = arg

    def add_dir_arg(self, name, tip="", default="", required=True):
        arg = DirectoryArg(self.value_arg_panel)
        arg.tip = tip or name.strip("-")
        arg.value = "" if required else default

        if required:
            self.positinal_args[name] = arg
        else:
            self.optional_args[name] = arg

    def add_boolean_arg(self, name, tip="", default=False):
        arg = BooleanArg(self.boolean_arg_panel)
        arg.tip = tip or name.strip("-")
        arg.value = default

        self.boolean_args[name] = arg

    def confirm(self):
        self._args.clear()
        for name, w in self.positinal_args.items():
            v = w.value
            if not v:
                messagebox.showerror("参数错误", f"{w.tip} 不能为空")
                self._args.clear()
                return None
            self._args.append(v)

        for name, w in self.optional_args.items():
            v = w.value
            if v:
                self._args.append(name)
                self._args.append(v)

        for name, w in self.boolean_args.items():
            v = w.value
            if v:
                self._args.append(name)

        self.root.destroy()

    def get_args(self) -> List[str]:
        self._init()
        self.root.mainloop()
        return self._args.copy()


if __name__ == "__main__":
    ui = ArgumentParserGUI()
    ui.add_dir_arg("dirchoo", "选则一个文件夹")
    ui.add_file_arg("--filechho", "选择一个文件", required=False)
    ui.add_value_arg("something", "输入任意之")
    ui.add_boolean_arg("--switch", "这是一个开关", default=True)
    ui.add_boolean_arg("--switch2", "这是一个开关", default=True)
    ui.add_boolean_arg("--switch3", "这是一个开关", default=True)
    ui.add_boolean_arg("--switch4", "这是一个开关", default=True)

    args = ui.get_args()

    print(args)
