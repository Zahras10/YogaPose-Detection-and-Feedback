

from tkinter import *
import tkinter as tk


from PIL import Image ,ImageTk

from tkinter.ttk import *
from pymsgbox import *


root=tk.Tk()

root.title("benefits Of Yoga Pose ")
w,h = root.winfo_screenwidth(),root.winfo_screenheight()

image2 =Image.open('y2.jpeg')
image2 =image2.resize((w,h), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)
#, relwidth=1, relheight=1)

w = tk.Label(root, background="Alice blue",foreground="Black", text="FIT-YOGA",width=20,height=2,font=("Marker Felt",25,"bold"))
w.place(x=15,y=15)

def vajrasan():
    
    
    label_l1 = tk.Label(root, text="___Benefits of vajrasan___\n 1.Good for our Digestive System.\n 2.Relieves Low Back Pain.\n 3.Keeps blood sugar levels under control.\n 4.Relieves Rheumatic Pain",font=("Times New Roman", 15, 'bold'),
                        background="black", fg="white", width=55, height=6)
    label_l1.place(x=950, y=10)

def Anjaneyasana():
    
    
    label_l1 = tk.Label(root, text=" ___Benefits of Anjaneyasana___ \nAnjaneyasana stretches multiple key muscles in the body\n required for peak athletic performance —including the hamstrings, \n hip flexors, psoas, back, groin, and neck. It especially\n useful for runners or people who engage in high-impact",font=("Times New Roman", 15, 'bold'),
                        background="black", fg="white", width=55, height=6)
    label_l1.place(x=950, y=160)

def Bhujangasana():
    
    
    label_l1 = tk.Label(root, text=" ___Benefits of Bhujangasana___ \n It may help to stretch muscles in the chest, shoulders and \n abdominal area. It may help to soothe sciatica. \n It may help to enhance flexibility. \n It may rejuvenate the heart.\n It may elevate the mood.",font=("Times New Roman", 15, 'bold'),
                        background="black", fg="white", width=55, height=6)
    label_l1.place(x=950, y=310)

def Dhanurasan():
    
    
    label_l1 = tk.Label(root, text=" ___Benefits of Dhanurasan___ \n Dhanurasana or bow pose is a complete Yoga Asana \n that helps to strengthen the back and abdominal muscles.\n The health benefits may include enhancing blood circulation, \n adjusting hunched back and body posture, managing diabetes.",font=("Times New Roman", 15, 'bold'),
                        background="black", fg="white", width=55, height=6)
    label_l1.place(x=950, y=460)
    
def Bhadrasan():
    
    
    label_l1 = tk.Label(root, text=" ___Benefits of Bhadrasan___ \n 1. Develops flexibility of legs.\n 2.Improves digestion.\n 3.Strengthens backbone, thighs, hips and buttocks. \n 4. Activates muladhara or root chakra. \n 5. Eases delivery, labour during childbirth.",font=("Times New Roman", 15, 'bold'),
                        background="black", fg="white", width=55, height=6)
    label_l1.place(x=950, y=610)



w,h = root.winfo_screenwidth(),root.winfo_screenheight()
root.geometry("%dx%d+0+0"%(w,h))
root.configure(background="skyblue")









d2=tk.Button(root,text="Vajrasan Benefits",command=vajrasan,width=30,height=2,bd=0,background="skyblue",foreground="black",font=("times new roman",14,"bold"))
d2.place(x=100,y=200)


d3=tk.Button(root,text="Anjaneyasana Benefits",command=Anjaneyasana,width=30,height=2,bd=0,background="skyblue",foreground="black",font=("times new roman",14,"bold"))
d3.place(x=100,y=300)

d3=tk.Button(root,text="Bhujangasana Benefits",command=Bhujangasana,width=30,height=2,bd=0,background="skyblue",foreground="black",font=("times new roman",14,"bold"))
d3.place(x=100,y=400)

d3=tk.Button(root,text="Dhanurasan Benefits",command=Dhanurasan,width=30,height=2,bd=0,background="skyblue",foreground="black",font=("times new roman",14,"bold"))
d3.place(x=100,y=500)

d3=tk.Button(root,text="Bhadrasan Benefits",command=Bhadrasan,width=30,height=2,bd=0,background="skyblue",foreground="black",font=("times new roman",14,"bold"))
d3.place(x=100,y=600)
root.mainloop()
