# Smart Attendance System using Face recognition

In this python project, I have made an attendance system which takes attendance by using face recognition technique. I have also intergrated it with GUI (Graphical user interface) so it can be easy to use by anyone. GUI for this project is also made on python using tkinter.

This smart attendance system is a technology-based solution that uses facial recognition algorithms to automate attendance tracking in various settings such as schools, universities, offices, and other organizations. The system captures an image of a person's face and matches it against a pre-existing database of images to record attendance automatically.

**How to Install:**

1 - Clone this repository to your local machine.

2 - Install the required dependencies by running the following command in the terminal:

 pip install -r requirements.txt
 
3 - Set up the facial recognition model by following the instructions in the HCD/README.md file.

4 - Run the application by running the following command in the terminal:

 python hcd.py

**How to Use:**

1 - Ensure that the camera is connected and working properly.

2 - Launch the application by running the hcd.py script.

3 - Register new users by clicking on the "For New Registration" button and providing the required information.

4 - To take attendance, click on the "Take Attendance" button and allow the camera to capture the faces of the individuals in the designated area.

5 - The attendance data will be stored in the folder of Attendance\Attendance_date.csv can be viewed on the attendance table.

**Dependencies:**

Python 3.7+

OpenCV

Numpy

Pandas

datetime

CSV

**Technology Used:**

1 - tkinter for whole GUI

2 - OpenCV for taking images and face recognition (cv2.face.LBPHFaceRecognizer_create())

**Features:**

Easy to use with interactive GUI support.

- Password protection for new person registration.

- Creates/Updates CSV file for details of students on registration.

- Creates a new CSV file everyday for attendance and marks attendance with proper date and time.

- Displays live attendance updates for the day on the main screen in tabular format with Id, name, date and time.

# Youtube link for reference

Youtube link: https://youtu.be/YVGM4aKMsTs

# SCREENSHOTS

Dashboard:
<img width="960" alt="image" src="https://user-images.githubusercontent.com/95516314/235356009-831b3dbf-e86a-4867-96a2-8832b3fe26ef.png">

For New Registration:
<img width="958" alt="image" src="https://user-images.githubusercontent.com/95516314/235356127-b0487869-ca63-4bea-87c5-99b00883ffc7.png">

Take Attendance:
<img width="957" alt="image" src="https://user-images.githubusercontent.com/95516314/235356208-68ffcab3-5459-4e60-90b7-6595610b711b.png">

Show Attendance taken:
<img width="960" alt="image" src="https://user-images.githubusercontent.com/95516314/235356254-f8c5b0eb-e3d0-4552-903f-b1a61803f222.png">

Help Menu:
<img width="960" alt="image" src="https://user-images.githubusercontent.com/95516314/235356283-e2483472-d231-4348-92ab-3983ce550c12.png">
<img width="960" alt="image" src="https://user-images.githubusercontent.com/95516314/235356304-06e2c46e-2a00-4f4d-980f-a486c255d2f6.png">






