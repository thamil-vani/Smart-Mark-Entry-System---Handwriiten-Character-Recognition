import tkinter
from tkinter.scrolledtext import ScrolledText
import os
from tkinter import *
import datetime as dt
import time
from PIL import ImageTk,Image
import sys
import cv2
from cv2 import *  

import tkinter.ttk as ttk
from tkinter import filedialog

import tkinter as ttk

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

#IMPORT SYS  #for multiline subject

import sqlite3

def quitting8():
        mnewscreen.destroy()
        manualscreen.destroy()
        manualmark()
   
    
def condb():
       
            conn = sqlite3.connect('stud.db')
            with conn:
                cursor=conn.cursor()
                
            query='UPDATE Student set CO1=? where rollno=?'
            query2='UPDATE Student set CO2=? where rollno=?'
            query3='UPDATE Student set TOTAL=? where rollno=?'
            
            mco1=qco1.get()
            mco2=qco2.get()
            rollnumber=qroll.get()
            totalmarks=float(mco1)+float(mco2)
            
            cursor.execute(query,(mco1,rollnumber,))
            cursor.execute(query2,(mco2,rollnumber,))
            cursor.execute(query3,(totalmarks,rollnumber,))
            conn.commit()            
            
       
            
       
            global mnewscreen
            mnewscreen=Toplevel(manualscreen)
            mnewscreen.title("Mark update")
            mnewscreen.geometry("300x300")
            
            label1=Label(mnewscreen,text="MARKS UPDATED",font=("Cambria",17,'bold')).pack()        
        
            btn1bg = Frame(mnewscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
            btn1=Button(btn1bg,text="CONTINUE", height="1", width="10",command=quitting8,font=("Cambria",17,"bold"),bg="paleturquoise")
            btn1bg.place(x=80,y=100)
            btn1.pack()
            mnewscreen.mainloop()
            
def manualmark1():
    imageprocessingscreen.destroy()
    manualmark()
    

def manualmark():
    
    
    global manualscreen
    manualscreen=Toplevel(loginsuccessscreen)
    manualscreen.title("Manual Entry")
    manualscreen.geometry("1000x820")

    label1=Label(manualscreen,text="SMART MARK ENTRY SYSTEM", fg="black",bg="darkturquoise", width="300", height="2", font=("Cambria", 24,'bold')).pack()
    label2=Label(manualscreen,text="MANUAL MARK ENTRY",fg="black",bg="antiquewhite",width="200",height="1",font=("Cambria",20,"bold")).pack()
 
    FILENAME ='bigimfs1.jpg'
    canvas = Canvas(manualscreen, width=1000, height=820)
    canvas.pack()
    tk_img = ImageTk.PhotoImage(file = FILENAME)
    canvas.create_image(500,334, image=tk_img)

    label3=Label(manualscreen,text="FILL THE DETAILS BELOW",fg="Black",font=("Cambria",17,"bold")).place(x=350,y=160)

    global qroll,qco1,qco2 
    
    qroll=StringVar()
    qco1=StringVar() 
    qco2=StringVar() 
    label4=Label(manualscreen,text="ROLL NUMBER ",font=('Cambria',15,'bold')).place(x=280,y=325)
    
    entrybg = Frame(manualscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    subjectentry=Entry(entrybg,width=30,font=('10'),textvar=qroll)
    entrybg.place(x=550,y=320)
    subjectentry.pack(ipady=4)
    
    label5=Label(manualscreen,text="CO-1 ",font=('Cambria',15,'bold')).place(x=280,y=425)
    
    entrybg = Frame(manualscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    markentry=Entry(entrybg,width=30,font=('10'),textvar=qco1)
    entrybg.place(x=550,y=420)
    markentry.pack(ipady=4)
    
    label6=Label(manualscreen,text="CO-2 ",font=('Cambria',15,'bold')).place(x=280,y=525)
    
    entrybg = Frame(manualscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    nmarkentry=Entry(entrybg,width=30,font=('10'),textvar=qco2)
    entrybg.place(x=550,y=520)
    nmarkentry.pack(ipady=4)
    
    btn1bg = Frame(manualscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn1=Button(btn1bg,text="CHANGE", height="1", width="10",command=condb,font=("Cambria",17,"bold"),bg="paleturquoise")
    btn1bg.place(x=350,y=600)
    btn1.pack()
    
    btn2bg = Frame(manualscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn2=Button(btn2bg,text="BACK", height="1", width="10",command=quitting7,font=("Cambria",17,"bold"),bg="paleturquoise")
    btn2bg.place(x=650,y=600)
    btn2.pack()
    
    
    
    manualscreen.mainloop()

def quitting7():
    manualscreen.destroy()
    
    
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import ImageFilter,Image




def detection():
    imageprocessingscreen.destroy()
    global model
    model = load_model('cnn_model_epoch10.h5')
    location()
    
    
def input_emnist(st):
        
        global co1,co2,finalrno,total
    	
        im_open = Image.open(st)
        im = Image.open(st).convert('LA') #conversion to gray-scale image
        width = float(im.size[0])
        height = float(im.size[1])
        newImage = Image.new('L',(28,28),(255))
    
    
        if width > height: #check which dimension is bigger
            #Width is bigger. Width becomes 20 pixels.
            nheight = int(round((28.0/width*height),0)) #resize height according to ratio width
            if (nheight == 0): #rare case but minimum is 1 pixel
                nheight = 1  
            # resize and sharpen
            img = im.resize((28,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
            newImage.paste(img, (0,wtop)) #paste resized image on white canvas
        else:
        #Height is bigger. Heigth becomes 20 pixels. 
            nwidth = int(round((28.0/height*width),0)) #resize width according to ratio height
            if (nwidth == 0): #rare case but minimum is 1 pixel
                nwidth = 1
         # resize and sharpen
            img = im.resize((nwidth,28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
            newImage.paste(img, (wleft,0)) #paste resize
    
    
    # # Normalizing image into pixel values
    
    
    
        tv = list(newImage.getdata())
        tva = [ (255-x)*1.0/255.0 for x in tv]
    
    
    
        for i in range(len(tva)):
            if tva[i]<=0.45:
                tva[i]=0.0
        n_image = np.array(tva)    
        rn_image = n_image.reshape(28,28)

        # return all the images
        
        return n_image,im_open,newImage
    
def recognition():
    
    from PIL import Image    
    
    imgfilename=imglocation.get()
    
    def locationcall2():
        locationscreen.destroy()
        filescreen.destroy()
        location()
    
    
    try:
        img=Image.open(imgfilename)
    except IOError:
        global filescreen
        filescreen=Toplevel(locationscreen)
        filescreen.title("File open error")
        filescreen.geometry("500x400")
        
        label1=Label(filescreen,text="File doesn't exist ",font=("Cambria",17,'bold')).pack()        
    
        btn1bg = Frame(filescreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
        btn1=Button(btn1bg,text="OK", height="1", width="10",command=locationcall2,font=("Cambria",17,"bold"),bg="paleturquoise")
        btn1bg.place(x=180,y=100)
        btn1.pack()        
        filescreen.mainloop()
    else:
        
        im = Image.open(imgfilename) 

        im1 = im.crop((27,1119,4616,1838)) 
        
        im2 = im.crop((6,2537,2348,3485)) 
        
        im3 = im.crop((2397,2558,4624,3476)) 
        
        
        im1.save(r'roll_number/rno.png')
        im2.save(r'co1/co1.png')
        im3.save(r'co2/co2.png')
        
        import cv2
        from imutils import contours
        
        # Load image1, grayscale, Gaussian blur, Canny edge detection
        image1 = cv2.imread(r"roll_number/rno.png")
        original = image1.copy()
        gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3,3), 0)
        canny = cv2.Canny(blurred, 120, 255, 1)
        
        # Find contours
        contour_list = []
        ROI_number = 0
        cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts, _ = contours.sort_contours(cnts, method="left-to-right")
        for c in cnts:
            # Obtain bounding rectangle for each contour
            x,y,w,h = cv2.boundingRect(c)
        
            # Find ROI of the contour
            roi = image1[y:y+h, x:x+w]
        
            # Draw bounding box rectangle, crop using Numpy slicing
            cv2.rectangle(image1,(x,y),(x+w,y+h),(0,255,0),3)
            ROI = original[y:y+h, x:x+w]
            cv2.imwrite(r'roll_number/ROI_r{}.png'.format(ROI_number), ROI)
            contour_list.append(c)
            ROI_number += 1
        
        print('Contours Detected: {}'.format(len(contour_list)))
        
        
        x=len(contour_list)
        
        
        
        l=[]
        global model
        model = load_model('cnn_model_epoch10.h5')
        
        
        
        for i in range(7):
            if(i!=2 and i!=3):
                image_2 = Image.open(r'roll_number/ROI_r{}.png'.format(i))
                x2 = image_2.resize((722, 992))
                x2.save(r'roll_number/img{}.png'.format(i))
                
                n_image,image,convo_image = input_emnist(r'roll_number/img{}.png'.format(i))
                img_class = model.predict(n_image.reshape(1,28,28,1))
                x=str(img_class.argmax())
                l.append(x)
        
        import pytesseract
        from PIL import Image, ImageEnhance, ImageFilter
        
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        
        im = Image.open(r'roll_number/ROI_r3.png') 
        
        im1 = im.crop((40,60,510,631)) 
        x2 = im1.resize((150,100))
        
        x2.save(r'roll_number/char.png')
        
        im = Image.open('roll_number/char.png')  # img is the path of the image 
        im = im.convert("RGBA")
        newimdata = []
        datas = im.getdata()
        
        for item in datas:
            if item[0] < 112 or item[1] < 112 or item[2] < 112:
                newimdata.append(item)
            else:
                newimdata.append((255, 255, 255))
        im.putdata(newimdata)
        
        im = im.filter(ImageFilter.MedianFilter())
        enhancer = ImageEnhance.Contrast(im)
        im = enhancer.enhance(2)
        im = im.convert('1')
        im.save(r'roll_number/temp2.jpg')
        text = pytesseract.image_to_string(Image.open(r'roll_number/temp2.jpg'),config='--psm 6', lang='eng')
        
        l.insert(2,str(text))
        roll=''.join(l)
        print(roll)
        
        
        
        image1 = cv2.imread(r"co1/co1.png")
        original = image1.copy()
        gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3,3), 0)
        canny = cv2.Canny(blurred, 120, 255, 1)
        
        # Find contours
        contour_list = []
        ROI_number = 0
        cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts, _ = contours.sort_contours(cnts, method="left-to-right")
        for c in cnts:
            # Obtain bounding rectangle for each contour
            x,y,w,h = cv2.boundingRect(c)
        
            # Find ROI of the contour
            roi = image1[y:y+h, x:x+w]
        
            # Draw bounding box rectangle, crop using Numpy slicing
            cv2.rectangle(image1,(x,y),(x+w,y+h),(0,255,0),3)
            ROI = original[y:y+h, x:x+w]
            cv2.imwrite(r'co1/ROI_r1{}.png'.format(ROI_number), ROI)
            contour_list.append(c)
            ROI_number += 1
        
        print('Contours Detected: {}'.format(len(contour_list)))
        
        
        x=len(contour_list)
        
        
        
        l2=[]
        
        
        for i in range(4):
            if(i!=2):
                image_2 = Image.open(r'co1/ROI_r1{}.png'.format(i))
                x2 = image_2.resize((722, 992))
                x2.save(r'co1/img1{}.png'.format(i))
                
                n_image,image,convo_image = input_emnist(r'co1/img1{}.png'.format(i))
                img_class = model.predict(n_image.reshape(1,28,28,1))
                x=str(img_class.argmax())
                l2.append(x)
        l2.insert(2,'.')
        co1final=''.join(l2)
        co1=float(co1final)
        print(co1)
        
        
        
        image1 = cv2.imread(r"co2/co2.png")
        original = image1.copy()
        gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3,3), 0)
        canny = cv2.Canny(blurred, 120, 255, 1)
        
        # Find contours
        contour_list = []
        ROI_number = 0
        cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts, _ = contours.sort_contours(cnts, method="left-to-right")
        for c in cnts:
            # Obtain bounding rectangle for each contour
            x,y,w,h = cv2.boundingRect(c)
        
            # Find ROI of the contour
            roi = image1[y:y+h, x:x+w]
        
            # Draw bounding box rectangle, crop using Numpy slicing
            cv2.rectangle(image1,(x,y),(x+w,y+h),(0,255,0),3)
            ROI = original[y:y+h, x:x+w]
            cv2.imwrite(r'co2/ROI_r2{}.png'.format(ROI_number), ROI)
            contour_list.append(c)
            ROI_number += 1
        
        print('Contours Detected: {}'.format(len(contour_list)))
        
        
        x=len(contour_list)
        
        
        
        l3=[]
        
        
        for i in range(4):
            if(i!=2):
                image_2 = Image.open(r'co2/ROI_r2{}.png'.format(i))
                x2 = image_2.resize((722, 992))
                x2.save(r'co2/img2{}.png'.format(i))
                
                n_image,image,convo_image = input_emnist(r'co2/img2{}.png'.format(i))
                img_class = model.predict(n_image.reshape(1,28,28,1))
                x=str(img_class.argmax())
                l3.append(x)
        
        l3.insert(2,'.')
        co2final=''.join(l3)
        co2=float(co2final)
        print(co2)
        
        total=co1+co2
        print(total)
        

        
        
        conn = sqlite3.connect('stud.db')
        with conn:
            cursor=conn.cursor()
        
        query='UPDATE Student set CO1=? where rollno=?'
        query2='UPDATE Student set CO2=? where rollno=?'
        query3='UPDATE Student set TOTAL=? where rollno=?'
        
        cursor.execute(query,(co1,roll,))
        cursor.execute(query2,(co2,roll,))
        cursor.execute(query3,(total,roll,))
        conn.commit()
        
        locationscreen.destroy()
        
        
        def quitting9():
            loadscreen.destroy()
            
        def locationcall():
            loadscreen.destroy()
            location()
        
        global loadscreen
        loadscreen=Toplevel(loginsuccessscreen)
        loadscreen.title("Loaded")
        loadscreen.geometry("500x400")
        
        label1=Label(loadscreen,text="Values detected and updated into the database",font=("Cambria",17,'bold')).pack()        
    
        btn1bg = Frame(loadscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
        btn1=Button(btn1bg,text="CONTINUE", height="1", width="10",command=locationcall,font=("Cambria",17,"bold"),bg="paleturquoise")
        btn1bg.place(x=180,y=100)
        btn1.pack()
        
        btn2bg = Frame(loadscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
        btn2=Button(btn2bg,text="QUIT ENTRY", height="1", width="10",command=quitting9,font=("Cambria",17,"bold"),bg="paleturquoise")
        btn2bg.place(x=180,y=200)
        btn2.pack()
        
        loadscreen.mainloop()



def location():

    global locationscreen
    locationscreen=Toplevel(loginsuccessscreen)

    locationscreen.title("Image Scanning")
    locationscreen.geometry("800x500")
    label1=Label(locationscreen,text="SMART MARK ENTRY SYSTEM", fg="black",bg="darkturquoise", width="300", height="2", font=("Cambria", 24,'bold')).pack()
   
    l2=Label(locationscreen,text="ENTER NAME OF THE IMAGE FILE",fg="black",bg="antiquewhite",width="200",height="1",font=("Cambria",17,"bold")).pack()
    
    FILENAME ='bigimfs1.jpg'
    canvas = Canvas(locationscreen, width=800, height=500)
    canvas.pack()
    tk_img = ImageTk.PhotoImage(file = FILENAME)
    canvas.create_image(500,334, image=tk_img)    
    
    label1=Label(locationscreen,text="FILENAME ",font=('Cambria',15,'bold')).place(x=150,y=225)
    
    global imglocation
    imglocation =StringVar()
    
    entrybg = Frame(locationscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    subjectentry=Entry(entrybg,width=30,font=('10'),textvar=imglocation)
    entrybg.place(x=400,y=220)
    subjectentry.pack(ipady=4)
    
    
      
    btn1bg = Frame(locationscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn1=Button(btn1bg,text="OK", height="1", width="10",command=recognition,font=("Cambria",17,"bold"),bg="paleturquoise")
    btn1bg.place(x=330,y=400)
    btn1.pack()

    locationscreen.mainloop()
    
def phonevideo():
        imageprocessingscreen.destroy()
        url = 'http://192.168.43.1:8080/video'
        cap = VideoCapture(url)
        
        while(True):
            ret, frame = cap.read()
            if frame is not None:
                imshow('frame',frame)
            q = cv2.waitKey(1)
            if q == ord("q"):
                break
            imwrite("phone.jpg",frame)
    
    
        destroyAllWindows()
        imshow("Scanned Image",frame)

                
        

def video():
        imageprocessingscreen.destroy()
        cam = VideoCapture(0) # 0 -> index of camera

        while True:
                s, img = cam.read()
                imshow("Scanner",img)
                
                if waitKey(1) & 0xFF == ord('q'):
                    break
       
                imwrite("filename.jpg",img)
                
        cam.release()
        destroyAllWindows()
        imshow("Scanned Image",img)
    
def ipback():
    imageprocessingscreen.destroy()
    
def imageprocessing():
    global imageprocessingscreen
    imageprocessingscreen=Toplevel(loginsuccessscreen)
    imageprocessingscreen.title("Image Scanning")
    imageprocessingscreen.geometry("1000x820")

    label1=Label(imageprocessingscreen,text="SMART MARK ENTRY SYSTEM", fg="black",bg="darkturquoise", width="300", height="2", font=("Cambria", 24,'bold')).pack()
    label2=Label(imageprocessingscreen,text="IMAGE INPUT",fg="black",bg="antiquewhite",width="200",height="1",font=("Cambria",20,"bold")).pack()
 
    FILENAME ='bigimfs1.jpg'
    canvas = Canvas(imageprocessingscreen, width=1000, height=820)
    canvas.pack()
    tk_img = ImageTk.PhotoImage(file = FILENAME)
    canvas.create_image(500,334, image=tk_img)    


    btn1bg = Frame(imageprocessingscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn1=Button(btn1bg,text="UPLOAD IMAGE", height="1", width="30",command=detection,font=("Cambria",15,"bold"),bg="powderblue")
    btn1bg.place(x=300,y=200)
    btn1.pack()
    
    btn3bg = Frame(imageprocessingscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn3=Button(btn3bg,text="MANUAL MARK ENTRY", height="1",command=manualmark1, width="30",font=("Cambria",15,"bold"),bg="powderblue")
    btn3bg.place(x=300,y=300)
    btn3.pack()
    
    btn3bg = Frame(imageprocessingscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn3=Button(btn3bg,text="SCAN THROUGH WEBCAM", height="1",command=video, width="30",font=("Cambria",15,"bold"),bg="powderblue")
    btn3bg.place(x=300,y=400)
    btn3.pack()

   

    btn3bg = Frame(imageprocessingscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn3=Button(btn3bg,text="SCAN THROUGH PHONE", height="1",command=phonevideo, width="30",font=("Cambria",15,"bold"),bg="powderblue")
    btn3bg.place(x=300,y=500)
    btn3.pack()
    
    btn2bg = Frame(imageprocessingscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn2=Button(btn2bg,text="BACK", height="1",command=ipback, width="30",font=("Cambria",15,"bold"),bg="powderblue")
    btn2bg.place(x=300,y=600)
    btn2.pack()
    
    imageprocessingscreen.mainloop()    
 


def quitting4():
    newscreen.destroy()
    gmailscreen.destroy()
    


def sendmail():
   
    mail_subject=subjectmail.get()
    mail_content=contententry.get(1.0, END)
    
        #The mail addresses and password
    sender_address = 'spdpython@gmail.com'
    sender_pass = 'spdpython2020'
    
        #Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    
    check=tkvar.get()
    if(check=='All'):
        receiver_address =list1
        message['To'] =", ".join(receiver_address)
    else:
        receiver_address=check
        message['To'] =receiver_address


    message['Subject'] = mail_subject
        #The subject line
            #The body and the attachments for the mail

    message.attach(MIMEText(mail_content, 'plain'))
  

    try:
        fname
    except NameError:
        session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
        session.starttls() #enable security
        session.login(sender_address,sender_pass) #login with mail_id and password
        session.sendmail(sender_address, receiver_address, message.as_string())
        session.quit()
        
        
        
        
    else:    
        attach_file_name = fname
        attach_file = open(attach_file_name, 'rb') # Open the file as binary mode
        payload = MIMEBase('application', 'octate-stream')
        payload.set_payload((attach_file).read())
        encoders.encode_base64(payload) #encode the attachment
                #add payload header with filename
        payload.add_header('Content-Disposition', u'attachment', filename=str(attach_file_name))
        message.attach(payload)
            #Create SMTP session for sending the mail

        session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
        session.starttls() #enable security
        session.login(sender_address,sender_pass) #login with mail_id and password
        session.sendmail(sender_address, receiver_address, message.as_string())
        session.quit()
    
    
    global newscreen
    newscreen=Toplevel(gmailscreen)
    newscreen.title("Mail Sent")
    newscreen.geometry("300x300")
    
    label1=Label(newscreen,text="Mail successfully sent",font=("Cambria",17,'bold')).pack()        

    btn1bg = Frame(newscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn1=Button(btn1bg,text="OK", height="1", width="10",command=quitting4,font=("Cambria",17,"bold"),bg="paleturquoise")
    btn1bg.place(x=80,y=100)
    btn1.pack()
    newscreen.mainloop()


def selectfile():
    global fname

    fname = filedialog.askopenfilenames(
    	parent=gmailscreen,
    	initialdir='/',
    	initialfile='tmp',
    	filetypes=[
            ("All files", "*"),
    		("PNG", "*.png"),
    		("JPEG", "*.jpg")
    		])
   
    fname=(os.path.basename(str(fname)))
    fname=fname[:-3]  
    
    label1=Label(gmailscreen,text=fname, fg="black", width="18", font=("Cambria", 15,'bold')).place(x=160,y=555)
    

def gmail():
    global list1
    
    conn = sqlite3.connect('stud.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM student")
    rows = cursor.fetchall()
    count=0
    list1=[]
    for row in rows:
        if(count==4):
            break
        list1.append(row[5])
        count=count+1
    
    
    global gmailscreen
    gmailscreen=Toplevel(loginsuccessscreen)
    gmailscreen.title("Notify through Mail")
    gmailscreen.geometry("1000x820")

    label1=Label(gmailscreen,text="SMART MARK ENTRY SYSTEM", fg="black",bg="darkturquoise", width="300", height="2", font=("Cambria", 24,'bold')).pack()
    label2=Label(gmailscreen,text="NOTIFY VIA EMAILS",fg="black",bg="antiquewhite",width="200",height="1",font=("Cambria",20,"bold")).pack()
 
    FILENAME ='bigimfs1.jpg'
    canvas = Canvas(gmailscreen, width=1000, height=820)
    canvas.pack()
    tk_img = ImageTk.PhotoImage(file = FILENAME)
    canvas.create_image(500,334, image=tk_img)

    label3=Label(gmailscreen,text="FILL THE DETAILS BELOW",fg="Black",font=("Cambria",17,"bold")).place(x=350,y=160)

    global subjectmail
    global contententry
    
    subjectmail=StringVar()
    contententry=StringVar()
        
    label4=Label(gmailscreen,text="SUBJECT ",font=('Cambria',15,'bold')).place(x=280,y=225)
    
    entrybg = Frame(gmailscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    subjectentry=Entry(entrybg,width=30,font=('10'),textvar=subjectmail)
    entrybg.place(x=550,y=220)
    subjectentry.pack(ipady=4)

    

    label5=Label(gmailscreen,text="CONTENTS",font=("Cambria",15,'bold')).place(x=280,y=325)
    
    entrybg = Frame(gmailscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)   
    contententry=ScrolledText(entrybg,width=30,height=5,font=('10'))
    entrybg.place(x=550,y=320)
    contententry.pack(ipady=4)    

    btn1bg = Frame(gmailscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn1=Button(btn1bg,text="SEND", height="1", width="10",command=sendmail,font=("Cambria",17,"bold"),bg="paleturquoise")
    btn1bg.place(x=600,y=500)
    btn1.pack()
    
    length=len(list1)


    choices=list1
    choices.append('All')
    global tkvar
    tkvar = StringVar()
    tkvar.set(choices[length])
    popupMenu = OptionMenu(gmailscreen, tkvar, *choices)
    popupMenu.place(x=800,y=510)
      
    btn2bg = Frame(gmailscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn2=Button(btn2bg,text="Select file", height="1", width="10",command=selectfile,font=("Cambria",17,"bold"),bg="paleturquoise")
    btn2bg.place(x=200,y=500)
    btn2.pack()
    
    gmailscreen.mainloop()   
    

                    
                    
def marks():
    conn = sqlite3.connect('stud.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM student")
    rows = cursor.fetchall()
    lst=[]
    a1=('ROLL NUMBER', 'NAME','CO1','CO2', 'TOTAL','EMAIL')
    lst.append(a1)
    for row in rows:
        lst.append(row)  
    class Table: 
    	
    	def __init__(self,root): 
    		# code for creating table 
    		for i in range(total_rows): 
    			for j in range(5): 
    				
    				self.e = Entry(root, width=15, fg='black',font=('Cambria',16,'bold')) 
    				
    				self.e.grid(row=i, column=j) 
    				self.e.insert(END, lst[i][j])  
    
    # find total number of rows and 
    # columns in list 
    total_rows = len(lst) 
    #total_columns = len(lst[0]) 
    
    # create root window 
    root = Tk() 
    root.title("Marks display")
    t = Table(root) 
    root.mainloop() 
    
    
def studentmarks():
    studpass=studentusernameverify.get() 
    conn = sqlite3.connect('stud.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM student")
    rows = cursor.fetchall()
    lst=[]
    
    a=('ROLL NUMBER', 'NAME','CO1','CO2', 'TOTAL','EMAIL')
    lst.append(a)
   
    for row in rows:
        if (studpass == row[0]):
            lst.append(row)
            
        
    class Table: 
    	
    	def __init__(self,root): 
    		
    		# code for creating table 
    		for i in range(2): 
    			for j in range(2,5): 
    				
    				self.e = Entry(root, width=20, fg='black', 
    							font=('Cambria',16,'bold')) 
                    
    				
    				self.e.grid(row=i, column=j) 
    				self.e.insert(END, lst[i][j])  
    
    # find total number of rows and 
    # columns in list 
   
   # total_rows = 1
    #total_columns = 2 
        
    
    # create root window 
    root = Tk() 
    root.title("Marks display")
    t = Table(root) 
    root.mainloop() 
    

def quitting12():
    snewscreen.destroy()
    studentgmailscreen.destroy()
    

    

    

def studentsendmail():
    
   
    student_mail_subject=studentsubjectmail.get()
    student_mail_content=studentcontententry.get(1.0, END)
    
        #The mail addresses and password
    sender_address = 'smartmarkentry@gmail.com'
    sender_pass = 'smartmarkentry2020'
    
        #Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    
    
    receiver_address = radd.get()
    message['To'] = receiver_address
   


    message['Subject'] = student_mail_subject
        #The subject line
            #The body and the attachments for the mail

    message.attach(MIMEText(student_mail_content, 'plain'))
    

  

    try:
        filname
       
    except NameError:
       
        session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
        session.starttls() #enable security
        session.login(sender_address,sender_pass) #login with mail_id and password
        session.sendmail(sender_address, receiver_address, message.as_string())
        session.quit()
        
        
        
        
    else:   
        attach_file_name = filname
        attach_file = open(attach_file_name, 'rb') # Open the file as binary mode
        payload = MIMEBase('application', 'octate-stream')
        payload.set_payload((attach_file).read())
        encoders.encode_base64(payload) #encode the attachment
                #add payload header with filename
        payload.add_header('Content-Disposition', u'attachment', filename=str(attach_file_name))
        message.attach(payload)
            #Create SMTP session for sending the mail

        session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
        session.starttls() #enable security
        session.login(sender_address,sender_pass) #login with mail_id and password
        session.sendmail(sender_address, receiver_address, message.as_string())
        session.quit()
        
    
        
    
    
    global snewscreen
    snewscreen=Toplevel(studentgmailscreen)
    snewscreen.title("Mail Sent")
    snewscreen.geometry("300x300")
    
    label1=Label(snewscreen,text="Mail successfully sent",font=("Cambria",17,'bold')).pack()        

    btn1bg = Frame(snewscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn1=Button(btn1bg,text="OK", height="1", width="10",command=quitting12,font=("Cambria",17,"bold"),bg="paleturquoise")
    btn1bg.place(x=80,y=100)
    btn1.pack()
    snewscreen.mainloop()

def studentselectfile():
    global filname

    filname = filedialog.askopenfilenames(
    	parent=studentgmailscreen,
    	initialdir='/',
    	initialfile='tmp',
    	filetypes=[
            ("All files", "*"),
    		("PNG", "*.png"),
    		("JPEG", "*.jpg")
    		])
   
    filname=(os.path.basename(str(filname)))
    filname=filname[:-3]  
    
    label1=Label(studentgmailscreen,text=filname, fg="black", width="18", font=("Cambria", 15,'bold')).place(x=160,y=555)
    filname1=filname


    
def studentgmail():

    
    global studentgmailscreen
    studentgmailscreen=Toplevel(studentscreen)
    studentgmailscreen.title("Notify through Mail")
    studentgmailscreen.geometry("1000x820")

    label1=Label(studentgmailscreen,text="SMART MARK ENTRY SYSTEM", fg="black",bg="darkturquoise", width="300", height="2", font=("Cambria", 24,'bold')).pack()
    label2=Label(studentgmailscreen,text="NOTIFY VIA EMAILS",fg="black",bg="antiquewhite",width="200",height="1",font=("Cambria",20,"bold")).pack()
 
    FILENAME ='bigimfs1.jpg'
    canvas = Canvas(studentgmailscreen, width=1000, height=820)
    canvas.pack()
    tk_img = ImageTk.PhotoImage(file = FILENAME)
    canvas.create_image(500,334, image=tk_img)

    

    global studentsubjectmail
    global studentcontententry
    global radd
    
    studentsubjectmail=StringVar()
    studentcontententry=StringVar()
    radd=StringVar()
    
    label5=Label(studentgmailscreen,text="RECEIVER ADDRESS ",font=('Cambria',15,'bold')).place(x=280,y=155)    
    entrybg = Frame(studentgmailscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    subjectentry=Entry(entrybg,width=30,font=('10'),textvar=radd)
    entrybg.place(x=550,y=160)
    subjectentry.pack(ipady=4)
        
    label4=Label(studentgmailscreen,text="SUBJECT ",font=('Cambria',15,'bold')).place(x=280,y=235)    
    entrybg = Frame(studentgmailscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    subjectentry=Entry(entrybg,width=30,font=('10'),textvar=studentsubjectmail)
    entrybg.place(x=550,y=240)
    subjectentry.pack(ipady=4)
    

    label5=Label(studentgmailscreen,text="CONTENTS",font=("Cambria",15,'bold')).place(x=280,y=328)    
    entrybg = Frame(studentgmailscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)   
    studentcontententry=ScrolledText(entrybg,width=30,height=5,font=('10'))
    entrybg.place(x=550,y=330)
    studentcontententry.pack(ipady=4)    
    
    
    

    btn1bg = Frame(studentgmailscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn1=Button(btn1bg,text="SEND", height="1", width="10",command=studentsendmail,font=("Cambria",17,"bold"),bg="paleturquoise")
    btn1bg.place(x=600,y=500)
    btn1.pack()   

      
    btn2bg = Frame(studentgmailscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn2=Button(btn2bg,text="Select file", height="1", width="10",command=studentselectfile,font=("Cambria",17,"bold"),bg="paleturquoise")
    btn2bg.place(x=200,y=500)
    btn2.pack()
    
    studentgmailscreen.mainloop()   
def studentloginsuccess():
    global studentscreen
    studentloginscreen.destroy()
    studentscreen=Tk()
    studentscreen.title('Student page')
    studentscreen.geometry("1000x820")
    
    label1=Label(studentscreen,text="SMART MARK ENTRY SYSTEM", fg="black",bg="darkturquoise", width="300", height="2", font=("Cambria", 24,'bold')).pack()
    label2=Label(studentscreen,text="STUDENT",fg="black",bg="antiquewhite",width="200",height="1",font=("Cambria",20,"bold")).pack()
    FILENAME ='bigimfs1.jpg'
    canvas = Canvas(studentscreen, width=1000, height=820)
    canvas.pack()
    tk_img = ImageTk.PhotoImage(file = FILENAME)
    canvas.create_image(500,334, image=tk_img)
    
    label2=Label(studentscreen,text="SUCCESSFULLY LOGGED",font=("Cambria",17,'bold')).place(x=360,y=150)
    
    btn2bg = Frame(studentscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn2=Button(btn2bg,text="VIEW MARKS", height="1", width="20",command=studentmarks,font=("Cambria",15,"bold"),bg="powderblue")
    btn2bg.place(x=350,y=350)
    btn2.pack()
    
    btn4bg = Frame(studentscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn4=Button(btn4bg,text="NOTIFY THROUGH MAIL", height="1", width="20",command=studentgmail,font=("Cambria",15,"bold"),bg="powderblue")
    btn4bg.place(x=350,y=450)
    btn4.pack()
    
    btn3bg = Frame(studentscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn3=Button(btn3bg,text="EXIT", height="1",command=quitting6, width="20",font=("Cambria",15,"bold"),bg="powderblue")
    btn3bg.place(x=350,y=550)
    btn3.pack()
    
    studentscreen.mainloop()

def ip():
    imageprocessing()
    

    
def loginsuccess():
    global loginsuccessscreen
    loginscreen.destroy()
    loginsuccessscreen=Tk()
    loginsuccessscreen.title("Success")
    loginsuccessscreen.geometry("1000x820")

    label1=Label(loginsuccessscreen,text="SMART MARK ENTRY SYSTEM", fg="black",bg="darkturquoise", width="300", height="2", font=("Cambria", 24,'bold')).pack()
    label2=Label(loginsuccessscreen,text="FACULTY",fg="black",bg="antiquewhite",width="200",height="1",font=("Cambria",20,"bold")).pack()
    FILENAME ='bigimfs1.jpg'
    canvas = Canvas(loginsuccessscreen, width=1000, height=820)
    canvas.pack()
    tk_img = ImageTk.PhotoImage(file = FILENAME)
    canvas.create_image(500,334, image=tk_img)
    
    label2=Label(loginsuccessscreen,text="SUCCESSFULLY LOGGED",font=("Cambria",17,'bold')).place(x=400,y=150)

    btn1bg = Frame(loginsuccessscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn1=Button(btn1bg,text="MARK ENTRY", height="2", width="20",command=ip,font=("Cambria",15,"bold"),bg="powderblue")
    btn1bg.place(x=200,y=350)
    btn1.pack()

    btn2bg = Frame(loginsuccessscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn2=Button(btn2bg,text="VIEW MARKS", height="2", width="20",command=marks,font=("Cambria",15,"bold"),bg="powderblue")
    btn2bg.place(x=200,y=450)
    btn2.pack()

    btn3bg = Frame(loginsuccessscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn3=Button(btn3bg,text="EXIT", height="2",command=quitting5, width="20",font=("Cambria",15,"bold"),bg="powderblue")
    btn3bg.place(x=600,y=450)
    btn3.pack()

    btn4bg = Frame(loginsuccessscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn4=Button(btn4bg,text="NOTIFY THROUGH MAIL", height="2", width="20",command=gmail,font=("Cambria",15,"bold"),bg="powderblue")
    btn4bg.place(x=600,y=350)
    btn4.pack()

    loginsuccessscreen.mainloop()
    
def passwordnot():
    passwordnotrecognisedscreen.destroy()
    
def studentpasswordnot():
    studentpasswordnotrecognisedscreen.destroy()
    
    
def deleteuser():
    usernotfoundscreen.destroy()
    
def studentdeleteuser():
    studentusernotfoundscreen.destroy()

def loginverify():    
    username1=usernameverify.get()
    password1=passwordverify.get()  

    list_of_files=os.listdir()    

    if (username1 in list_of_files):
        file1=open(username1,"r")
        verify=file1.read()
        if password1 in verify:
            loginsuccess()
        
        else:
            global passwordnotrecognisedscreen
            passwordnotrecognisedscreen=Toplevel(loginscreen)
            passwordnotrecognisedscreen.title("Failure")
            passwordnotrecognisedscreen.geometry("250x150")
            label1=Label(passwordnotrecognisedscreen,text="Invalid Password").pack()
    
            btn1=Button(passwordnotrecognisedscreen,text="OK",command=passwordnot).pack() 
            passwordnotrecognisedscreen.mainloop()
            

            
    else:
           global usernotfoundscreen
           usernotfoundscreen = Toplevel(loginscreen)
           usernotfoundscreen.title("Success")
           usernotfoundscreen.geometry("250x150")
           label1=Label(usernotfoundscreen, text="User Not Found").pack()
           
           btn1=Button(usernotfoundscreen,text="OK",command=deleteuser).pack()
           usernotfoundscreen.mainloop()
           
def studentloginverify():    
    studentusername1=studentusernameverify.get()
    studentpassword1=studentpasswordverify.get()  
    #print(studentusername1,studentpassword1)

    list_of_files1=os.listdir()    

    if (studentusername1 in list_of_files1):
        file1=open(studentusername1,"r")
        verify=file1.read()
        if studentpassword1 in verify:
            studentloginsuccess()
        
        else:
            global studentpasswordnotrecognisedscreen
            studentpasswordnotrecognisedscreen=Toplevel(studentloginscreen)
            studentpasswordnotrecognisedscreen.title("Failure")
            studentpasswordnotrecognisedscreen.geometry("250x150")
            label1=Label(studentpasswordnotrecognisedscreen,text="Invalid Password").pack()
    
            btn1=Button(studentpasswordnotrecognisedscreen,text="OK",command=studentpasswordnot).pack() 
            studentpasswordnotrecognisedscreen.mainloop()
            

            
    else:
           global studentusernotfoundscreen
           studentusernotfoundscreen = Toplevel(studentloginscreen)
           studentusernotfoundscreen.title("Success")
           studentusernotfoundscreen.geometry("250x150")
           label1=Label(studentusernotfoundscreen, text="User Not Found").pack()
           
           btn1=Button(studentusernotfoundscreen,text="OK",command=studentdeleteuser).pack()
           studentusernotfoundscreen.mainloop()

def alphanumeric(char):
            return char.isalpha() or char.isdigit()


def studentlogin():
    global studentloginscreen
    userscreen.destroy()
    studentloginscreen=Tk()
    studentloginscreen.title("Login")
    studentloginscreen.geometry("1000x820")
    
    label1=Label(studentloginscreen,text="SMART MARK ENTRY SYSTEM", fg="black",bg="darkturquoise", width="300", height="2", font=("Cambria", 24,'bold')).pack()
    label2=Label(studentloginscreen,text="LOGIN",fg="black",bg="antiquewhite",width="200",height="1",font=("Cambria",20,"bold")).pack()
 
    FILENAME ='bigimfs1.jpg'
    canvas = Canvas(studentloginscreen, width=1000, height=820)
    canvas.pack()
    tk_img = ImageTk.PhotoImage(file = FILENAME)
    canvas.create_image(500,334, image=tk_img)   

    label3=Label(studentloginscreen,text="ENTER THE DETAILS BELOW",fg="Black",font=("Cambria",17,"bold")).place(x=350,y=180)

    global studentusernameverify
    global studentpasswordverify
    studentusernameverify=StringVar()
    studentpasswordverify=StringVar()
        
    label4=Label(studentloginscreen,text="ROLL NUMBER ",font=('Cambria',15,'bold')).place(x=280,y=345)
    
    entrybg = Frame(studentloginscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    validation=studentloginscreen.register(alphanumeric)
    studentusernameloginentry=Entry(entrybg,width=20,font=('10'),textvar=studentusernameverify,validate="key", validatecommand=(validation,'%S'))
    entrybg.place(x=550,y=340)
    studentusernameloginentry.pack(ipady=4)

    label5=Label(studentloginscreen,text="PASSWORD ",font=("Cambria",15,'bold')).place(x=280,y=445)
    
    entrybg = Frame(studentloginscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)   
    studentpasswordloginentry=Entry(entrybg,width=20,font=('10'),textvariable=studentpasswordverify, show= '*')
    entrybg.place(x=550,y=440)
    studentpasswordloginentry.pack(ipady=4)

    btn1bg = Frame(studentloginscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn1=Button(btn1bg,text="LOGIN", height="1", width="20",command=studentloginverify,font=("Cambria",17,"bold"),bg="paleturquoise")
    btn1bg.place(x=400,y=550)
    btn1.pack()

    studentloginscreen.mainloop()



def login():
    global loginscreen
    mainscreen.destroy()
    loginscreen=Tk()
    loginscreen.title("Login")
    loginscreen.geometry("1000x820")
    
    label1=Label(loginscreen,text="SMART MARK ENTRY SYSTEM", fg="black",bg="darkturquoise", width="300", height="2", font=("Cambria", 24,'bold')).pack()
    label2=Label(loginscreen,text="LOGIN",fg="black",bg="antiquewhite",width="200",height="1",font=("Cambria",20,"bold")).pack()
 
    FILENAME ='bigimfs1.jpg'
    canvas = Canvas(loginscreen, width=1000, height=820)
    canvas.pack()
    tk_img = ImageTk.PhotoImage(file = FILENAME)
    canvas.create_image(500,334, image=tk_img)   

    label3=Label(loginscreen,text="ENTER THE DETAILS BELOW",fg="Black",font=("Cambria",17,"bold")).place(x=350,y=180)

    global usernameverify
    global passwordverify
    usernameverify=StringVar()
    passwordverify=StringVar()
        
    label4=Label(loginscreen,text="USERNAME ",font=('Cambria',15,'bold')).place(x=280,y=345)
    
    entrybg = Frame(loginscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    validation=loginscreen.register(alphanumeric)
    usernameloginentry=Entry(entrybg,width=20,font=('10'),textvar=usernameverify,validate="key", validatecommand=(validation,'%S'))
    entrybg.place(x=550,y=340)
    usernameloginentry.pack(ipady=4)

    label5=Label(loginscreen,text="PASSWORD ",font=("Cambria",15,'bold')).place(x=280,y=445)
    
    entrybg = Frame(loginscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)   
    passwordloginentry=Entry(entrybg,width=20,font=('10'),textvariable=passwordverify, show= '*')
    entrybg.place(x=550,y=440)
    passwordloginentry.pack(ipady=4)

    btn1bg = Frame(loginscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn1=Button(btn1bg,text="LOGIN", height="1", width="20",command=loginverify,font=("Cambria",17,"bold"),bg="paleturquoise")
    btn1bg.place(x=400,y=550)
    btn1.pack()

    loginscreen.mainloop()

def deleteregister():
    registersuccessscreen.destroy()
     

def registersuccess():
    registerscreen.destroy()
    mainscreen.destroy()
    
    global registersuccessscreen
    registersuccessscreen=Tk()

    registersuccessscreen.title("Success")
    registersuccessscreen.geometry("300x300")
    label1=Label(registersuccessscreen,text="Registration Success",font=("Cambria",17,'bold')).pack()    

    btn1bg = Frame(registersuccessscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn1=Button(btn1bg,text="CONTINUE", height="1", width="10",command=deleteregister,font=("Cambria",17,"bold"),bg="paleturquoise")
    btn1bg.place(x=80,y=100)
    btn1.pack()
    
    
def registerfailure():
        global registerfailurescreen
        mainscreen.destroy()
        registerfailurescreen=Tk()
        registerfailurescreen.title("Failure")
        registerfailurescreen.geometry("300x300")
        
        label1=Label(registerfailurescreen,text="Passwords don't match!!!\n  Try again",font=("Cambria",17,'bold')).pack()        

        btn1bg = Frame(registerfailurescreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
        btn1=Button(btn1bg,text="CONTINUE", height="1", width="10",command=registerfailurescreen.destroy,font=("Cambria",17,"bold"),bg="paleturquoise")
        btn1bg.place(x=80,y=100)
        btn1.pack()

def registeruser():
    usernameinfo=username.get()
    passwordinfo=password.get()
    passwordcheck=passwordconfirm.get()

    if(passwordinfo==passwordcheck):
            file=open(usernameinfo,"w")
            file.write(passwordinfo)
            file.close()
            registersuccess()

    else:
            registerfailure()
    
def register():
    global username
    global password
    global passwordconfirm
    global usernameentry
    global passwordentry
    global registerscreen
    registerscreen=Toplevel(mainscreen)
    registerscreen.title("Register")
    registerscreen.geometry("1000x820")

    label1=Label(registerscreen,text="SMART MARK ENTRY SYSTEM", fg="black",bg="darkturquoise", width="300", height="2", font=("Cambria", 24,'bold')).pack()
    label2=Label(registerscreen,text="REGISTER",fg="black",bg="antiquewhite",width="200",height="1",font=("Cambria",20,"bold")).pack()

    FILENAME ='bigimfs1.jpg'
    canvas = Canvas(registerscreen, width=1000, height=820)
    canvas.pack()
    tk_img = ImageTk.PhotoImage(file = FILENAME)
    canvas.create_image(500,334, image=tk_img)
    
    label3=Label(registerscreen, text="Please enter details below",font=("Cambria",17,'bold')).place(x=350,y=125)     
    
    usernamelabel=Label(registerscreen,text="USERNAME ",font=("Cambria",17,'bold'))
    usernamelabel.place(x=250,y=295)    
    
    username=StringVar()
    password=StringVar()
    passwordconfirm=StringVar()

    entrybg = Frame(registerscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    validation=registerscreen.register(alphanumeric)
    usernameentry=Entry(entrybg,width=20,font=('10'),textvar=username,validate="key", validatecommand=(validation,'%S'))
    entrybg.place(x=600,y=290)
    usernameentry.pack(ipady=4)
    
    passwordlabel=Label(registerscreen,text="PASSWORD ",font=("Cambria",17,'bold'))
    passwordlabel.place(x=250,y=395)

    entrybg = Frame(registerscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    validation=registerscreen.register(alphanumeric)
    passwordentry=Entry(entrybg,width=20,font=('10'),textvar=password,show="*",validate="key", validatecommand=(validation,'%S'))
    entrybg.place(x=600,y=390)
    passwordentry.pack(ipady=4)

    passwordlabel=Label(registerscreen,text="CONFIRM PASSWORD ",font=("Cambria",17,'bold'))
    passwordlabel.place(x=250,y=495)

    entrybg = Frame(registerscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    validation=registerscreen.register(alphanumeric)
    passwordentry=Entry(entrybg,width=20,font=('10'),textvar=passwordconfirm,show="*",validate="key", validatecommand=(validation,'%S'))
    entrybg.place(x=600,y=490)
    passwordentry.pack(ipady=4)    

    btn1bg = Frame(registerscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn1=Button(btn1bg,text="SUBMIT", height="1", width="10",command=registeruser,font=("Cambria",17,"bold"),bg="paleturquoise")
    btn1bg.place(x=450,y=570)
    btn1.pack()

    registerscreen.mainloop()  


def quitting5():
        loginsuccessscreen.destroy()
        sys.exit('user quitted')
        
def quitting6():
    studentscreen.destroy()
    sys.exit('User quitted')
        
        
def quitting2():
        mainscreen.destroy()        
        

def quitting3():
        userscreen.destroy()        
        sys.exit('user quitted')

def mainaccountscreen():
    userscreen.destroy()
    global mainscreen
    mainscreen=Tk()
    mainscreen.geometry("1000x820")
    mainscreen.title("Account Login")
   

    label1=Label(mainscreen,text="SMART MARK ENTRY SYSTEM", fg="black",bg="darkturquoise", width="300", height="2", font=("Cambria", 24,'bold')).pack()

    w = Label(mainscreen, text=f"{dt.datetime.now():%a, %b %d %Y}", fg="black", bg="antiquewhite",width=300,height=1, font=("Cambria", 18,'bold')).pack()

    FILENAME ='bigimfs1.jpg'
    canvas = Canvas(mainscreen, width=1000, height=667)
    canvas.pack()
    tk_img = ImageTk.PhotoImage(file = FILENAME)
    canvas.create_image(500,334, image=tk_img)
    
    btn1bg = Frame(mainscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn1=Button(btn1bg,text="LOGIN", height="1", width="20",command=login,font=("Cambria",17,"bold"),bg="paleturquoise")
    btn1bg.place(x=350,y=350)
    btn1.pack()
     
        
    btn2bg = Frame(mainscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn2=Button(btn2bg,text="REGISTER", height="1", width="20",command=register,font=("Cambria",17,"bold"),bg="paleturquoise")
    btn2bg.place(x=350,y=450)
    btn2.pack()

    btn3bg = Frame(mainscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn3=Button(btn3bg,text="BACK", height="1", width="20",command=quitting2,font=("Cambria",17,"bold"),bg="paleturquoise")
    btn3bg.place(x=350,y=550)
    btn3.pack()    
    
    mainscreen.mainloop()


def userselectionscreen():
    global userscreen
    userscreen=Tk()
    userscreen.geometry("1000x820")
    userscreen.title("Account Selection")
   

    label1=Label(userscreen,text="SMART MARK ENTRY SYSTEM", fg="black",bg="darkturquoise", width="300", height="2", font=("Cambria", 24,'bold')).pack()

    w = Label(userscreen, text=f"{dt.datetime.now():%a, %b %d %Y}", fg="black", bg="antiquewhite",width=300,height=1, font=("Cambria", 18,'bold')).pack()

    FILENAME ='bigimfs1.jpg'
    canvas = Canvas(userscreen, width=1000, height=667)
    canvas.pack()
    tk_img = ImageTk.PhotoImage(file = FILENAME)
    canvas.create_image(500,334, image=tk_img)
    
    btn1bg = Frame(userscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn1=Button(btn1bg,text="FACULTY", height="1", width="20",command=mainaccountscreen,font=("Cambria",17,"bold"),bg="paleturquoise")
    btn1bg.place(x=350,y=250)
    btn1.pack()
     
        
    btn2bg = Frame(userscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn2=Button(btn2bg,text="STUDENT", height="1", width="20",command=studentlogin,font=("Cambria",17,"bold"),bg="paleturquoise")
    btn2bg.place(x=350,y=350)
    btn2.pack()

    btn3bg = Frame(userscreen, background = 'BLACK', borderwidth = 3, relief = FLAT)
    btn3=Button(btn3bg,text="QUIT", height="1", width="20",command=quitting3,font=("Cambria",17,"bold"),bg="paleturquoise")
    btn3bg.place(x=350,y=450)
    btn3.pack()        
    
    userscreen.mainloop()
    

CHOICE=1
while(CHOICE==1):
       userselectionscreen()
       destroyAllWindows()
