import win32api
import win32gui
import win32con
import win32process
import ctypes
from math import ceil


def MAKELPARAM(lo, hi):
    return (int(lo) & 0xffff) | ((int(hi) & 0xffff) << 16)


def ismine(pt):
    """是否有雷"""
    return bool(pt & 0x80)


def ismined(pt):
    """是否挖开"""
    return bool(pt & 0x40)


def isflag(pt):
    """是否标旗"""
    return (pt & 0x0f) == 0x0e


def ismark(pt):
    """是否问号"""
    return (pt & 0x0f) == 0x0d


def isbomb(pt):
    """是否是触发地雷"""
    return (pt & 0x0f) == 0x0c


def isfakemine(pt):
    """是否是错误标旗"""
    return (pt & 0x0f) == 0x0b


def ischainbomb(pt):
    """是否被连锁引爆"""
    return (pt & 0x0f) == 0x0a


def getptnum(pt):
    """获取格子里的数字"""
    return pt & 0x0f


class WinMineCtl:
    init_x = 20
    init_y = 60
    unit = 16
    addr_status = 0x1005160  # 0: 进行中 1: 尝试中 2: 死亡 3: 胜利
    addr_row_num = 0x10056A8  # [9, 24]
    addr_col_num = 0x10056AC  # [9, 30]
    addr_mine_num = 0x10056A4  # [10, ceil(row*0.9)*ceil(col*0.9)]
    addr_mine_map = 0x1005340  # 有一圈边框

    def __init__(self, hwnd, scale=1):
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        if pid == -1:
            raise Exception("获取进程ID失败")
        self.handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, pid)
        if self.handle is None:
            raise Exception("获取进程句柄失败")
        self.hwnd = hwnd
        self.scale = scale

    @property
    def row_num(self):
        """行数"""
        return int.from_bytes(
            win32process.ReadProcessMemory(self.handle, self.addr_row_num, 4),
            "little", signed=True
        )

    @row_num.setter
    def row_num(self, value):
        win32process.WriteProcessMemory(
            self.handle, self.addr_row_num,
            int(max(9, min(value, 24))).to_bytes(4, "little", signed=True)
        )

    @property
    def col_num(self):
        """列数"""
        return int.from_bytes(
            win32process.ReadProcessMemory(self.handle, self.addr_col_num, 4),
            "little", signed=True
        )

    @col_num.setter
    def col_num(self, value):
        win32process.WriteProcessMemory(
            self.handle, self.addr_col_num,
            int(max(9, min(value, 30))).to_bytes(4, "little", signed=True)
        )

    @property
    def mine_num(self):
        """雷数"""
        return int.from_bytes(
            win32process.ReadProcessMemory(self.handle, self.addr_mine_num, 4),
            "little", signed=True
        )

    @mine_num.setter
    def mine_num(self, value):
        win32process.WriteProcessMemory(
            self.handle, self.addr_mine_num,
            int(max(10, min(value, ceil(self.row_num*0.9)*ceil(self.col_num*0.9)))).
            to_bytes(4, "little", signed=True)
        )

    @property
    def status(self):
        """
            0: 进行中 
            1: 尝试中 
            2: 死亡 
            3: 胜利
        """
        return int.from_bytes(
            win32process.ReadProcessMemory(self.handle, self.addr_status, 4),
            "little", signed=False
        )

    def getpt(self, row, col):
        """
            状态 个数
            0000 0000
            高位
            1000 有雷
            0100 挖开

            低位
            1111 无
            1110 旗
            1101 问号
            1100 触雷
            1011 误判
            1010 连锁雷
            0000-1000 个数
        """
        if row > 0 and col > 0 and row <= self.row_num and col <= self.col_num:
            return int.from_bytes(
                win32process.ReadProcessMemory(
                    self.handle,
                    self.addr_mine_map+row*32+col,
                    1
                ), "little", signed=False
            )
        return 0

    def transpt(self, row, col):
        return (
            (self.init_x+(col-1)*self.unit)*self.scale,
            (self.init_y+(row-1)*self.unit)*self.scale
        )

    def leftclk(self, row, col):
        if row > 0 and col > 0 and row <= self.row_num and col <= self.col_num:
            ptx, pty = self.transpt(row, col)
            win32gui.PostMessage(self.hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, MAKELPARAM(ptx, pty))
            win32gui.PostMessage(self.hwnd, win32con.WM_LBUTTONUP, 0, MAKELPARAM(ptx, pty))

    def rightclk(self, row, col):
        if row > 0 and col > 0 and row <= self.row_num and col <= self.col_num:
            ptx, pty = self.transpt(row, col)
            win32gui.PostMessage(self.hwnd, win32con.WM_RBUTTONDOWN, win32con.MK_RBUTTON, MAKELPARAM(ptx, pty))
            win32gui.PostMessage(self.hwnd, win32con.WM_RBUTTONUP, 0, MAKELPARAM(ptx, pty))

    def bothclk(self, row, col):
        if row > 0 and col > 0 and row <= self.row_num and col <= self.col_num:
            ptx, pty = self.transpt(row, col)
            win32gui.PostMessage(self.hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, MAKELPARAM(ptx, pty))

            win32gui.PostMessage(self.hwnd, win32con.WM_RBUTTONDOWN, win32con.MK_LBUTTON | win32con.MK_RBUTTON, MAKELPARAM(ptx, pty))
            win32gui.PostMessage(self.hwnd, win32con.WM_MOUSEMOVE, win32con.MK_LBUTTON | win32con.MK_RBUTTON, MAKELPARAM(ptx, pty))
            win32gui.PostMessage(self.hwnd, win32con.WM_RBUTTONUP, win32con.MK_LBUTTON, MAKELPARAM(ptx, pty))
            win32gui.PostMessage(self.hwnd, win32con.WM_MOUSEMOVE, win32con.MK_LBUTTON, MAKELPARAM(ptx, pty))

            win32gui.PostMessage(self.hwnd, win32con.WM_LBUTTONUP, 0, MAKELPARAM(ptx, pty))

    def newgame(self, row_num=None, col_num=None, mine_num=None):
        if row_num:
            self.row_num = row_num
        if col_num:
            self.col_num = col_num
        if mine_num:
            self.mine_num = mine_num

        win32gui.PostMessage(self.hwnd, win32con.WM_COMMAND, 0x101FE, 0)



if __name__ == "__main__":
    hwnd = win32gui.FindWindow(None, "扫雷")
    mctl = WinMineCtl(hwnd)

    print("行数: {} 列数: {} 雷数: {}".format(mctl.row_num, mctl.col_num, mctl.mine_num))
    print(mctl.status)
    mctl.leftclk(1, 1)
    mctl.rightclk(5, 5)
    print(mctl.status)