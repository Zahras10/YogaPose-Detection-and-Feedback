import tkinter as tk
from tkinter import ttk, LEFT, END
from tkinter import messagebox as ms
import sqlite3
from PIL import Image, ImageTk
import re


##############################################+=============================================================
root = tk.Tk()
root.configure(background="grey")
# root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Login Form")




username = tk.StringVar()
password = tk.StringVar()
        

# ++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 = Image.open('new6.jpg')
image2 = image2.resize((w,h), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)



def registration():
    from subprocess import call
    call(["python","registration.py"])
    root.destroy()

def login():
        # Establish Connection

    with sqlite3.connect('evaluation.db') as db:
         c = db.cursor()

        # Find user If there is any take proper action
         db = sqlite3.connect('evaluation.db')
         cursor = db.cursor()
         cursor.execute("CREATE TABLE IF NOT EXISTS registration"
                           "(Fullname TEXT, address TEXT, username TEXT, Email TEXT, Phoneno TEXT,Gender TEXT,age TEXT , password TEXT)")
         db.commit()
         find_entry = ('SELECT * FROM registration WHERE username = ? and password = ?')
         c.execute(find_entry, [(username.get()), (password.get())])
         result = c.fetchall()

         if result:
            msg = ""
            # self.logf.pack_forget()
            # self.head['text'] = self.username.get() + '\n Loged In'
            # msg = self.head['text']
            #            self.head['pady'] = 150
            print(msg)
            ms.showinfo("messege", "LogIn sucessfully")
            # ===========================================
            root.destroy()

           
            from subprocess import call
            call(['python','Home.py'])
            # ================================================
         else:
           ms.showerror('Oops!', 'Username Or Password Did Not Found/Match.')



title=tk.Label(root, text="Login Here", font=("Times", 30, "bold"),bd=10,fg="black",bg="Alice blue")
title.place(x=600,y=190,width=250)
        
Login_frame=tk.Frame(root,bg="grey")
Login_frame.place(x=500,y=300)
        
logolbl=tk.Label(Login_frame,bd=0).grid(row=0,columnspan=2,pady=20)
        
lbluser=tk.Label(Login_frame,text="Username",compound=LEFT,font=("Times new roman", 20, "bold"),bg="grey",fg="white").grid(row=1,column=0,padx=20,pady=10)
txtuser=tk.Entry(Login_frame,bd=2,textvariable=username,font=("",15))
txtuser.grid(row=1,column=1,padx=20)
        
lblpass=tk.Label(Login_frame,text="Password",compound=LEFT,font=("Times new roman", 20, "bold"),bg="grey",fg="white").grid(row=2,column=0,padx=50,pady=10)
txtpass=tk.Entry(Login_frame,bd=2,textvariable=password,show="*",font=("",15))
txtpass.grid(row=2,column=1,padx=20)

btn_log=tk.Button(Login_frame,text="Login",command=login,width=15,height=2,font=("Times new roman", 14, "bold"),fg="black")
btn_log.grid(row=3,column=1,pady=30)
btn_reg=tk.Button(Login_frame,text="Create Account",command=registration,width=15,height=2,font=("Times new roman", 14, "bold"),fg="black")
btn_reg.grid(row=3,column=0,pady=30)
        
       

root.mainloop()