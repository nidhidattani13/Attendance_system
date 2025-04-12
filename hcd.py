import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200, tick)


def contact():
    mess._show(title='Contact us', message="Please contact us on : 'kothariyash360@gmail.com' ")


def check_haarcascadefile():
    exists = os.path.isfile("haarcascade_frontalface_default.xml")
    if exists:
        pass
    else:
        mess._show(title='Some file missing', message='Please download haarcascade_frontalface_default.xml')
        window.destroy()


def save_pass():
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel/psd.txt")
    if exists1:
        tf = open("TrainingImageLabel/psd.txt", "r")
        key = tf.read()
    else:
        master.destroy()
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas == None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
        else:
            tf = open("TrainingImageLabel/psd.txt", "w")
            tf.write(new_pas)
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return
    op = (old.get())
    newp = (new.get())
    nnewp = (nnew.get())
    if (op == key):
        if(newp == nnewp):
            txf = open("TrainingImageLabel/psd.txt", "w")
            txf.write(newp)
        else:
            mess._show(title='Error', message='Confirm new password again!!!')
            return
    else:
        mess._show(title='Wrong Password', message='Please enter correct old password.')
        return
    mess._show(title='Password Changed', message='Password changed successfully!!')
    master.destroy()


def change_pass():
    global master
    master = tk.Tk()
    master.geometry("400x160")
    master.resizable(False, False)
    master.title("Change Password")
    master.configure(background="white")
    lbl4 = tk.Label(master, text='    Enter Old Password', bg='white', font=('times', 12, ' bold '))
    lbl4.place(x=10, y=10)
    global old
    old = tk.Entry(master, width=25, fg="black", relief='solid', font=('times', 12, ' bold '), show='*')
    old.place(x=180, y=10)
    lbl5 = tk.Label(master, text='   Enter New Password', bg='white', font=('times', 12, ' bold '))
    lbl5.place(x=10, y=45)
    global new
    new = tk.Entry(master, width=25, fg="black", relief='solid', font=('times', 12, ' bold '), show='*')
    new.place(x=180, y=45)
    lbl6 = tk.Label(master, text='Confirm New Password', bg='white', font=('times', 12, ' bold '))
    lbl6.place(x=10, y=80)
    global nnew
    nnew = tk.Entry(master, width=25, fg="black", relief='solid', font=('times', 12, ' bold '), show='*')
    nnew.place(x=180, y=80)
    cancel = tk.Button(master, text="Cancel", command=master.destroy, fg="black", bg="red", height=1, width=25, activebackground="white", font=('times', 10, ' bold '))
    cancel.place(x=200, y=120)
    save1 = tk.Button(master, text="Save", command=save_pass, fg="black", bg="#3ece48", height=1, width=25, activebackground="white", font=('times', 10, ' bold '))
    save1.place(x=10, y=120)
    master.mainloop()


def psw():
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel/psd.txt")
    if exists1:
        tf = open("TrainingImageLabel/psd.txt", "r")
        key = tf.read()
    else:
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas == None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
        else:
            tf = open("TrainingImageLabel/psd.txt", "w")
            tf.write(new_pas)
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return
    password = tsd.askstring('Password', 'Enter Password', show='*')
    if (password == key):
        TrainImages()
    elif (password == None):
        pass
    else:
        mess._show(title='Wrong Password', message='You have entered wrong password')


def clear():
    txt.delete(0, 'end')
    res = "First Click on Take Images then Save Profile"
    message1.configure(text=res)


def clear2():
    txt2.delete(0, 'end')
    res = "First Click on Take Images then Save Profile"
    message1.configure(text=res)


def TakeImages():
    check_haarcascadefile()
    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")
    serial = 0
    exists = os.path.isfile("StudentDetails/StudentDetails.csv")
    if exists:
        with open("StudentDetails/StudentDetails.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial = serial + 1
        serial = (serial // 2)
        csvFile1.close()
    else:
        with open("StudentDetails/StudentDetails.csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(columns)
            serial = 1
        csvFile1.close()
    Id = (txt.get())
    name = (txt2.get())
    
    # Improved name validation
    if any(c.isalpha() for c in name) and all(c.isalpha() or c.isspace() for c in name):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        
        # Display instructions
        print("Instructions for student:")
        print("1. Look straight at the camera")
        print("2. Slowly move your head slightly in different directions")
        print("3. Try different expressions (neutral, smile)")
        print("4. Press 'q' to stop capture early")
        
        while True:
            ret, img = cam.read()
            if not ret:
                mess._show(title='Camera Error', message='Failed to access camera. Please check connections.')
                break
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Improved face detection parameters
            faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            
            # Add feedback on face detection
            if len(faces) > 0:
                cv2.putText(img, f"Face detected! Capturing: {sampleNum}/100", (30, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(img, "No face detected! Please adjust position", (30, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add instructions on screen
            instruction_text = "Please move your face slightly. Press 'q' to stop."
            cv2.putText(img, instruction_text, (10, img.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum = sampleNum + 1
                
                # Save the captured face
                cv2.imwrite(f"TrainingImage/{name}.{str(serial)}.{Id}.{str(sampleNum)}.jpg",
                            gray[y:y + h, x:x + w])
            
            cv2.imshow('Taking Images', img)
            
            # Break if 'q' is pressed or enough samples are collected
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum >= 100:
                break
        
        cam.release()
        cv2.destroyAllWindows()
        
        # Update database
        res = f"Images Taken for ID: {Id} | Name: {name}"
        row = [serial, '', Id, '', name]
        with open('StudentDetails/StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message1.configure(text=res)
        
        # Show confirmation
        mess._show(title='Success', message=f'Captured {sampleNum} images for {name}')
    else:
        res = "Enter Correct name (only alphabets and spaces allowed)"
        message.configure(text=res)
        mess._show(title='Invalid Name', message='Name should contain only alphabets and spaces')


def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    
    # Check if training images exist
    if not os.listdir("TrainingImage/"):
        mess._show(title='No Data', message='Training images not found. Please capture images first!')
        return
        
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    
    # Show training progress
    progress_window = tk.Toplevel()
    progress_window.title("Training in progress")
    progress_window.geometry("300x100")
    progress_label = tk.Label(progress_window, text="Training model with images...\nPlease wait...", font=('times', 12))
    progress_label.pack(pady=20)
    progress_window.update()
    
    try:
        faces, ID = getImagesAndLabels("TrainingImage")
        recognizer.train(faces, np.array(ID))
        recognizer.save("TrainingImageLabel/Trainner.yml")
        
        # Close progress window and show success
        progress_window.destroy()
        res = "Profile Saved Successfully"
        message1.configure(text=res)
        message.configure(text='Total Registrations till now  : ' + str(ID[0]))
        mess._show(title='Training Complete', message='Model training completed successfully!')
    except Exception as e:
        progress_window.destroy()
        mess._show(title='Error', message=f'Training error: {str(e)}\nPlease check images and try again.')
        return


def getImagesAndLabels(path):
    # Get all file paths
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    
    # Show progress counter
    total_images = len(imagePaths)
    processed = 0

    for imagePath in imagePaths:
        try:
            # Load and convert to grayscale
            pilImage = Image.open(imagePath).convert('L')
            # Convert to numpy array
            imageNp = np.array(pilImage, 'uint8')
            # Get ID from filename
            ID = int(os.path.split(imagePath)[-1].split(".")[1])
            
            # Add to training data
            faces.append(imageNp)
            Ids.append(ID)
            
            # Update progress
            processed += 1
            if processed % 10 == 0:
                print(f"Processing images: {processed}/{total_images}")
                
        except Exception as e:
            print(f"Error processing {imagePath}: {e}")
            continue
            
    print(f"Processed {len(faces)} faces for training")
    return faces, Ids


def TrackImages(close_after_match=False):
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")
    
    # Clear existing data in the treeview
    for k in tv.get_children():
        tv.delete(k)
        
    msg = ''
    i = 0
    j = 0
    
    # Load the trained model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    exists3 = os.path.isfile("TrainingImageLabel/Trainner.yml")
    if exists3:
        recognizer.read("TrainingImageLabel/Trainner.yml")
    else:
        mess._show(title='Data Missing', message='Please click on Save Profile to train the model first!')
        return
        
    # Load face detector
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)

    # Start camera
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Set countdown parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = 1
    text_color = (0, 0, 255)
    start_time = time.time()
    countdown_duration = 3
    
    # Load student data
    col_names = ['Id', '', 'Name', '', 'Date', '', 'Time']
    exists1 = os.path.isfile("StudentDetails/StudentDetails.csv")
    if exists1:
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
    else:
        mess._show(title='Details Missing', message='Students details are missing, please check!')
        cam.release()
        cv2.destroyAllWindows()
        window.destroy()
        
    # Store recognized students to avoid duplicates in one session
    recognized_students = set()
    
    # Main recognition loop
    while True:
        ret, im = cam.read()
        if not ret:
            mess._show(title='Camera Error', message='Failed to access camera feed')
            break
            
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        current_time = time.time()
        time_remaining = countdown_duration - int(current_time - start_time)
        
        # Show countdown
        if time_remaining > 0:
            text = f"Starting in {time_remaining} seconds..."
            cv2.putText(im, text, (25, 25), font, text_size, text_color, 2)
            cv2.imshow('Taking Attendance', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Improved face detection
        faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        
        # Process detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            
            # Predict identity
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
            
            # Adjust confidence threshold as needed (lower = stricter)
            if conf < 85:  # Changed from 100 to 85 for stricter matching
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                
                # Get name from database
                try:
                    aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                    ID = df.loc[df['SERIAL NO.'] == serial]['ID'].values
                    ID = str(ID)
                    ID = ID[1:-1]
                    bb = str(aa)
                    bb = bb[2:-2]
                    
                    # Only record attendance once per session
                    student_key = f"{ID}_{bb}"
                    if student_key not in recognized_students:
                        attendance = [str(ID), '', bb, '', str(date), '', str(timeStamp)]
                        recognized_students.add(student_key)
                    if close_after_match and len(recognized_students) > 0:
                        mess._show(title='Match Found', 
                        message=f'Successfully identified: {bb}')
                        break
                        
                        # Show confirmation on screen
                        cv2.putText(im, f"Identified: {bb}", (x, y-10), font, 0.75, (0, 255, 0), 2)
                        
                        # Record attendance
                        ts = time.time()
                        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                        exists = os.path.isfile(f"Attendance/Attendance_{date}.csv")
                        
                        if exists:
                            with open(f"Attendance/Attendance_{date}.csv", 'a+') as csvFile1:
                                writer = csv.writer(csvFile1)
                                writer.writerow(attendance)
                            csvFile1.close()
                        else:
                            with open(f"Attendance/Attendance_{date}.csv", 'a+') as csvFile1:
                                writer = csv.writer(csvFile1)
                                writer.writerow(col_names)
                                writer.writerow(attendance)
                            csvFile1.close()
                            
                        # Play confirmation sound (optional)
                        # import winsound
                        # winsound.Beep(1000, 100)
                except Exception as e:
                    print(f"Error processing recognition: {e}")
                    bb = "Unknown"
            else:
                Id = 'Unknown'
                bb = str(Id)
                cv2.putText(im, "Unknown", (x, y-10), font, 0.75, (0, 0, 255), 2)
                
            # Display name
            cv2.putText(im, str(bb), (x, y + h), font, 1, (255, 255, 255), 2)
            
            # Display confidence level
            confidence_text = f"Match: {round(100 - conf)}%"
            cv2.putText(im, confidence_text, (x, y + h + 30), font, 0.6, (255, 200, 0), 1)
            
        # Add instructions
        cv2.putText(im, "Press 'q' to exit", (im.shape[1]-200, 30), font, 0.7, (0, 255, 255), 2)
        
        # Show the image
        cv2.imshow('Taking Attendance', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cam.release()
    cv2.destroyAllWindows()
    
    # Update attendance display
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    attendance_file = f"Attendance/Attendance_{date}.csv"
    
    # Check if attendance was recorded today
    if os.path.isfile(attendance_file):
        # Load attendance data into the treeview
        with open(attendance_file, 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for lines in reader1:
                i = i + 1
                if (i > 1):  # Skip header
                    if len(lines) > 0:
                        iidd = str(lines[0]) + '   '
                        tv.insert('', 0, text=iidd, values=(str(lines[2]), str(lines[4]), str(lines[6])))
        csvFile1.close()
        
        # Show completion message
        mess._show(title='Attendance Taken', 
                   message=f'Attendance recorded successfully for {len(recognized_students)} students!')
    else:
        mess._show(title='No Attendance', message='No attendance was recorded!')


# Initialize global variables
global key
key = ''

# Get current date
ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day, month, year = date.split("-")

# Month dictionary
mont = {'01': 'January',
        '02': 'February',
        '03': 'March',
        '04': 'April',
        '05': 'May',
        '06': 'June',
        '07': 'July',
        '08': 'August',
        '09': 'September',
        '10': 'October',
        '11': 'November',
        '12': 'December'
        }

# Create the main window
window = tk.Tk()
window.geometry("1280x720")
window.resizable(True, False)
window.title("Smart Attendance System")
window.configure(background='#EAEAEA')

# Create main frames
frame1 = tk.Frame(window, bg="#C1CDCD")
frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)

frame2 = tk.Frame(window, bg="#C1CDCD")
frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

# Title
message3 = tk.Label(window, text="Smart Attendance System", fg="white", bg="#262523", width=55, height=1, font=('times', 29, ' bold '))
message3.place(x=10, y=10)

# Date and time frames
frame3 = tk.Frame(window, bg="#c4c6ce")
frame3.place(relx=0.52, rely=0.09, relwidth=0.09, relheight=0.07)

frame4 = tk.Frame(window, bg="#c4c6ce")
frame4.place(relx=0.36, rely=0.09, relwidth=0.16, relheight=0.07)

# Date display
datef = tk.Label(frame4, text=day + "-" + mont[month] + "-" + year + " | ", fg="orange", bg="#262523", width=55, height=1, font=('times', 22, ' bold '))
datef.pack(fill='both', expand=1)

# Clock display
clock = tk.Label(frame3, fg="orange", bg="#262523", width=95, height=1, font=('times', 22, ' bold '))
clock.pack(fill='both', expand=1)
tick()

# Section headers
head2 = tk.Label(frame2, text="                       For New Registrations                       ", fg="black", bg="#EEE5DE", font=('times', 17, ' bold '))
head2.grid(row=0, column=0)

head1 = tk.Label(frame1, text="                       For Already Registered                       ", fg="black", bg="#EEE5DE", font=('times', 17, ' bold '))
head1.place(x=0, y=0)

# ID input
lbl = tk.Label(frame2, text="Enter ID", width=20, height=1, fg="black", bg="#CD7054", font=('times', 17, ' bold '))
lbl.place(x=80, y=55)

txt = tk.Entry(frame2, width=32, fg="black", font=('times', 15, ' bold '))
txt.place(x=30, y=88)

# Name input
lbl2 = tk.Label(frame2, text="Enter Name", width=20, fg="black", bg="#CD7054", font=('times', 17, ' bold '))
lbl2.place(x=80, y=140)

txt2 = tk.Entry(frame2, width=32, fg="black", font=('times', 15, ' bold '))
txt2.place(x=30, y=173)

# Instructions and status messages
message1 = tk.Label(frame2, text="First Click on Take Images then Save Profile", fg="black", width=39, height=1, activebackground="yellow", font=('times', 15, ' bold '))
message1.place(x=7, y=230)

message = tk.Label(frame2, text="", bg="#CD7054", fg="black", width=39, height=1, activebackground="yellow", font=('times', 16, ' bold '))
message.place(x=7, y=450)

lbl3 = tk.Label(frame1, text="Attendance", width=20, fg="black", bg="#CD7054", height=1, font=('times', 17, ' bold '))
lbl3.place(x=100, y=115)

# Count total registrations
res = 0
exists = os.path.isfile("StudentDetails/StudentDetails.csv")
if exists:
    with open("StudentDetails/StudentDetails.csv", 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        for l in reader1:
            res = res + 1
    res = (res // 2) - 1
    csvFile1.close()
else:
    res = 0
message.configure(text='Total Registrations till now  : ' + str(res))

# Menu bar
menubar = tk.Menu(window, relief='ridge')
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label='Change Password', command=change_pass)
filemenu.add_command(label='Contact Us', command=contact)
filemenu.add_command(label='Exit', command=window.destroy)
menubar.add_cascade(label='Help', font=('times', 29, ' bold '), menu=filemenu)

# Attendance view
tv = ttk.Treeview(frame1, height=13, columns=('name', 'date', 'time'))
tv.column('#0', width=82)
tv.column('name', width=130)
tv.column('date', width=133)
tv.column('time', width=133)
tv.grid(row=2, column=0, padx=(0, 0), pady=(150, 0), columnspan=4)
tv.heading('#0', text='ID')
tv.heading('name', text='NAME')
tv.heading('date', text='DATE')
tv.heading('time', text='TIME')

# Scrollbar for attendance view
scroll = ttk.Scrollbar(frame1, orient='vertical', command=tv.yview)
scroll.grid(row=2, column=4, padx=(0, 100), pady=(150, 0), sticky='ns')
tv.configure(yscrollcommand=scroll.set)

# Buttons
clearButton = tk.Button(frame2, text="Clear", command=clear, fg="black", bg="grey", width=11, activebackground="white", font=('times', 11, ' bold '))
clearButton.place(x=335, y=86)

clearButton2 = tk.Button(frame2, text="Clear", command=clear2, fg="black", bg="grey", width=11, activebackground="white", font=('times', 11, ' bold '))
clearButton2.place(x=335, y=172)

takeImg = tk.Button(frame2, text="Take Images", command=TakeImages, fg="black", bg="grey", width=14, height=1, activebackground="white", font=('times', 15, ' bold '))
takeImg.place(x=150, y=300)

trainImg = tk.Button(frame2, text="Save Profile", command=psw, fg="black", bg="grey", width=14, height=1, activebackground="white", font=('times', 15, ' bold '))
trainImg.place(x=150, y=380)

trackImg = tk.Button(frame1, text="Take Attendance", command=TrackImages, fg="black", bg="#a8ad8b", width=14, height=1, activebackground="white", font=('times', 15, ' bold '))
trackImg.place(x=150, y=50)

quitWindow = tk.Button(frame1, text="Quit", command=window.destroy, fg="black", bg="#e05561", width=14, height=1, activebackground="white", font=('times', 15, ' bold '))
quitWindow.place(x=150, y=450)

# Configure menu
window.configure(menu=menubar)

# Start the main loop
window.mainloop()