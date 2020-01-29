from ctypes import *
libc = cdll.LoadLibrary("main/libmain.so")
w = libc.__init_subclass__()
print(dir(w))
print(w)