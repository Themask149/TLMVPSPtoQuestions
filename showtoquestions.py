import numpy as np
import cv2
import pytesseract
import os
import sys
import contextlib
import time
os.environ['AV_LOG_LEVEL'] = 'error'

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

NAGUI=0
BOCCO=1
JARRY=2

class QuestionDetector:
    def __init__(self, video_path,mode):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.speed = self.frame_rate*5  # Speed factor for frame skipping
        self.current_frame=None
        self.current_frame_id=0
        self.video_title = os.path.splitext(os.path.basename(video_path))[0]
        self.questions =[]
        self.themes=[]
        self.mode=mode
        # Convert RGB to BGR
        self.orange_hsv = ((4,213,231),(24,233,251))
        self.green_hsv = ((50,120,245),(80,210,255))
        self.rose_hsv=((140,230,223),(176,255,250))
        self.violet_hsv=((110,150,220),(131,170,255))
        self.full_path=os.path.join( r'E:\Emissions\tlmvpsp',self.video_title)
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)
            print(f"Folder created: {self.full_path}")
        else:
            print(f"Folder already exists: {self.full_path}")
    
    def extract_question_from_frame_manche_1(self):
        height,width,channels = self.current_frame.shape

        cropped_img = self.current_frame[(85*height//100):(95*height//100),width//6:5*width//6,:]
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        _, thresholded_img = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
        question = pytesseract.image_to_string(thresholded_img, lang='fra')
        print(repr(question))
        self.questions.append(question.replace('\n',' '))

    def extract_question_from_frame_manche_2(self):
        height,width,channels = self.current_frame.shape

        cropped_img = self.current_frame[(80*height//100):(89*height//100),width//6:5*width//6,:]
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        _, thresholded_img = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
        question = pytesseract.image_to_string(thresholded_img, lang='fra')
        print(repr(question))
        self.questions.append(question.replace('\n',' '))

    def extract_theme(self):
        height,width,channels = self.current_frame.shape

        cropped_img = self.current_frame[(90*height//100):(96*height//100),35*width//100:65*width//100,:]
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        _, thresholded_img = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        theme = pytesseract.image_to_string(thresholded_img, lang='fra')
        print(repr(theme))
        self.themes.append(theme.replace("\n",""))

    def analyze_color(self, frame, target_color_bgr, tolerance=10):
        height, width = frame.shape[:2]
        cropped_frame = cv2.resize(frame[height//2:height, width//6:(width-width//6)], (width//8, height//8))
        cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

        # Define a range around the target color
        lower_bound,upper_bound=target_color_bgr

        # Create a mask for the target color
        mask = cv2.inRange(cropped_frame, lower_bound, upper_bound)

        # Calculate the ratio of target color pixels
        target_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        color_ratio = (target_pixels / total_pixels) * 100
        print(color_ratio)
        return color_ratio

    def look_for_color(self, color, threshold=0.001):
        while self.current_frame_id < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_id)
            ret, self.current_frame = self.cap.read()
            if ret:
                analyze_color = self.analyze_color(self.current_frame, color)
                if analyze_color > threshold and ( color!=self.orange_hsv or self.analyze_color(self.current_frame,self.green_hsv)<1 ):
                    print(f"{analyze_color}% for {color}")
                    print(f"frame_id:{self.current_frame_id} => time:{(self.current_frame_id//self.frame_rate)//60} min {(self.current_frame_id//self.frame_rate)%60 }")
                    self.current_frame_id += self.frame_rate
                    return
                self.current_frame_id += self.speed
    
    def look_for_color_and_show(self,color,threshold=0.001):
        t= time.time_ns()
        self.look_for_color(color,threshold)
        print(f"Temps pour une question {(time.time_ns()-t)/1000000} ms")
        if self.current_frame is not None:
            self.current_frame=cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
            cv2.imshow("frame", self.current_frame)
            cv2.waitKey(0)  # Wait indefinitely for a key press
            cv2.destroyAllWindows()  # Close the window
        return

    def find_one_question_round_1(self):
        print("orange")
        self.look_for_color(self.orange_hsv,1)
        self.speed=self.frame_rate//3
        print("green")
        self.look_for_color(self.green_hsv,2)
        self.speed=self.frame_rate*5
        self.extract_question_from_frame_manche_1()
    
    def find_all_questions_round_1_and_save(self):
        i=0
        if (self.mode!=NAGUI):
            self.find_one_question_round_1()
            filename=os.path.join(self.full_path,f'Question-Manche1-{i:03}.jpg')
            cv2.imwrite(filename,self.current_frame)
            i+=1
        for j in range(6):
            self.current_frame_id+=40*self.frame_rate
            self.find_one_question_round_1()
            filename=os.path.join(self.full_path,f'Question-Manche1-{i:03}.jpg')
            cv2.imwrite(filename,self.current_frame)
            i+=1
            self.find_one_question_round_1()
            filename=os.path.join(self.full_path,f'Question-Manche1-{i:03}.jpg')
            cv2.imwrite(filename,self.current_frame)
            i+=1
        if (self.mode!=NAGUI):
            self.find_one_question_round_1()
            filename=os.path.join(self.full_path,f'Question-Manche1-{i:03}.jpg')
            cv2.imwrite(filename,self.current_frame)
    
    def find_one_question_round_2(self):
        if (self.mode!=NAGUI):
            print("rose")
            self.look_for_color(self.rose_hsv,0.5)
            self.speed=self.frame_rate
            print("green")
            self.look_for_color(self.green_hsv,2)
            self.speed=self.frame_rate*5
            self.extract_question_from_frame_manche_2()
    
    def find_all_questions_round_2_and_save(self):
        nb=0
        if (self.mode==JARRY):
            nb=10
        else:
            nb=12
        for i in range(nb):
            self.find_one_question_round_2()
            self.current_frame_id+=10*self.frame_rate
            filename=os.path.join(self.full_path,f'Question-Manche2-{i:03}.jpg')
            cv2.imwrite(filename,self.current_frame)
        self.extract_theme()
        

    
    def find_one_question_round_3_champion(self):
        if (self.mode!=NAGUI):
            print("rose")
            self.look_for_color(self.rose_hsv,1)
            self.speed=self.frame_rate//2
            print("green")
            self.look_for_color(self.green_hsv,2.01)
            self.speed=self.frame_rate*5
            self.extract_question_from_frame_manche_2()

    def find_one_question_round_3_challenger(self):
        if (self.mode!=NAGUI):
            print("violet")
            self.look_for_color(self.violet_hsv,0.5)
            self.speed=self.frame_rate//2
            print("green")
            self.look_for_color(self.green_hsv,2.01)
            self.speed=self.frame_rate*5
            self.extract_question_from_frame_manche_2()
    
    def find_all_questions_round_3_and_save(self):
        nb=0
        if (self.mode==JARRY):
            nb=5
        else:
            nb=6
        for i in range(nb):
            self.find_one_question_round_3_champion()
            self.current_frame_id+=5*self.frame_rate
            filename=os.path.join(self.full_path,f'Question-Manche3-{i:03}.jpg')
            cv2.imwrite(filename,self.current_frame)
        self.extract_theme()
        
        for i in range(nb):
            self.find_one_question_round_3_challenger()
            self.current_frame_id+=5*self.frame_rate
            filename=os.path.join(self.full_path,f'Question-Manche3-{6+i:03}.jpg')
            cv2.imwrite(filename,self.current_frame)
        self.extract_theme()
        
    
    def run(self):
        self.current_frame_id=36*60*self.frame_rate
        self.find_all_questions_round_1_and_save()
        self.current_frame_id+=1*60*self.frame_rate
        self.find_all_questions_round_2_and_save()
        self.current_frame_id+=5*60*self.frame_rate
        self.find_all_questions_round_3_and_save()
        fullname=os.path.join(self.full_path,"questions.txt")
        with open(fullname, 'w') as file:
            for string in self.questions:
                file.write(string + '\n')
        fullname=os.path.join(self.full_path,"themes.txt")
        with open(fullname, 'w') as file:
            for string in self.themes:
                file.write(string + '\n')

    
    def __del__(self):
        # Destructor to release video capture resources
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("Video resources released.")

# Usage

if __name__=="__main__":
    if len(sys.argv)==3:
        video_path=sys.argv[1]
        mode = sys.argv[2]
        if (mode=="JARRY"):
            mode=JARRY
        elif (mode=="NAGUI"):
            mode=NAGUI
        elif (mode=="BOCCO"):
            mode=BOCCO
        else:
            print("JARRY mode by default")
            mode=JARRY
        question_detector = QuestionDetector(video_path,mode)
        question_detector.run()
       
    else:
        print("Usage: python showtoquestions.py <video_path> <mode>")
        print("Example: python showtoquestions.py 'E:\Emissions\Tout le monde veut prendre sa place_France 2_2024_01_04_11_57.mp4' JARRY")