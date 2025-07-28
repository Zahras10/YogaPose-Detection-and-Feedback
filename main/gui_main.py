
from tkinter import *
import tkinter as tk


from PIL import Image ,ImageTk

from tkinter.ttk import *
from pymsgbox import *


root=tk.Tk()

root.title("Yoga Pose Detection")
w,h = root.winfo_screenwidth(),root.winfo_screenheight()

image2 =Image.open('newname.jpg')
image2 =image2.resize((w,h), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)
#, relwidth=1, relheight=1)

w = tk.Label(root,background="skyblue",foreground="Black", text="FIT-YOGA",width=20,height=2,font=("Marker Felt",25,"bold"))
w.place(x=15,y=15)




w,h = root.winfo_screenwidth(),root.winfo_screenheight()
root.geometry("%dx%d+0+0"%(w,h))
root.configure(background="Alice Blue")


from tkinter import messagebox as ms


def Login():
    from subprocess import call
    call(["python","login1.py"])
def Register():
    from subprocess import call
    call(["python","registration.py"])


wlcm=tk.Label(root,text="Welcome to Yoga Pose Detection System",width=95,height=3,background="Alice Blue",foreground="black",font=("Times new roman",22,"bold"))
wlcm.place(x=0,y=root.winfo_screenheight()-150)



d2=tk.Button(root,text="Login",command=Login,width=15,height=3,bd=0,background="skyblue",foreground="black",font=("times new roman",16,"bold"))
d2.place(x=1000,y=15)


d3=tk.Button(root,text="Register",command=Register,width=15,height=3,bd=0,background="skyblue",foreground="black",font=("times new roman",16,"bold"))
d3.place(x=1200,y=15)



root.mainloop()
