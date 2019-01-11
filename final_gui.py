
# coding: utf-8

# In[4]:


from tkinter import *
import pyttsx3
import os
import sys
import subprocess

window = Tk()
"""filename = PhotoImage(file = "blind11.png")
background_label = Label(image=filename)
background_label.grid(row=0, column=0)
"""
window.title("Welcome to Our Blind Assistive TooL")

 
window.geometry('600x800')


window.configure(background='#0f0423')
lbl = Label(window, text="Press Q to Capture Image", font=("Arial Bold", 15))
lbl.grid(column=0, row=0)
lbl.place(relx=0.5,y=45,anchor="center")
#lbl.place(x=100,y=45,anchor="center")
engine=pyttsx3.init()
engine.say("Press Q to Capture Image")
engine.runAndWait()


def clicked():
 
    os.system('python camera.py')
    lbl1 = Label(window, text="Thankyou For Capturing !!", font=("Arial Bold", 15))
    lbl1.grid(column=0, row=3)
    engine=pyttsx3.init()
    engine.say("Thankyou For Capturing")
    engine.runAndWait()
    
btn = Button(window, text="Capture Image", bg="#bca703", fg="black", command=clicked,height = 2, width = 12,font = ('Arial','16','bold'))
btn.grid(column=1, row=1)
btn.place(relx=0.5,y=100,anchor="center")

def compute():
    engine=pyttsx3.init()
    #engine.say("wait For the Processing")
    engine.say("wait For the Processing")
    engine.runAndWait()
    cmd = 'python anushri.py'

    # it will execute script which runs only `function1`
    output = subprocess.check_output(cmd, shell=True)

    lbl['text'] = output.strip()
    
    #lbl1 = Label(window, text="Processing done", font=("Arial Bold", 15))
    #lbl1.grid(column=2, row=6)
    lbl.place(relx=0.5,y=450,anchor="center")
    



lbl1 = Label(window, text="Click process for image and object detection", font=("Arial Bold", 15))
lbl1.grid(column=0, row=4)
lbl1.place(relx=0.5,y=180,anchor="center")

btn1 = Button(window, text="Process", bg="#bca703", fg="black", command=compute,height = 2, width = 12, font = ('Arial','16','bold'))
btn1.grid(column=1, row=5)
btn1.place(relx=0.5,y=240,anchor="center")
 
window.mainloop()


# ## 
