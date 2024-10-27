import numpy as np
import cv2
import pytesseract
import os
import sys
import contextlib
import time
from dotenv import load_dotenv
import webhook
import requests
from bs4 import BeautifulSoup
import yt_dlp
import re
import numpy as np
import time
os.environ['AV_LOG_LEVEL'] = 'error'

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
download_folder="E:\Emissions\\"

NAGUI=0
BOCCO=1
JARRY=2
CYRIL=3
DEBUG=True

"""
format_id: dash-video=400000, ext: mp4, resolution: 384x216, note: DASH video
format_id: hls-522, ext: mp4, resolution: 384x216, note: N/A
format_id: dash-video=950000, ext: mp4, resolution: 640x360, note: DASH video
format_id: hls-1105, ext: mp4, resolution: 640x360, note: N/A
format_id: dash-video=1400000, ext: mp4, resolution: 960x540, note: DASH video
format_id: hls-1582, ext: mp4, resolution: 960x540, note: N/A
format_id: dash-video=2000000, ext: mp4, resolution: 1280x720, note: DASH video
format_id: hls-2218, ext: mp4, resolution: 1280x720, note: N/A
format_id: dash-video=5000000, ext: mp4, resolution: 1920x1080, note: DASH video
format_id: hls-5398, ext: mp4, resolution: 1920x1080, note: N/A
"""

class VideoDownloader:
    def __init__(self, source, pattern):
        self.source = source
        self.pattern = pattern
        self.urls = self.get_video_urls()
    
    def get_output_name(self,url):
        regex = r"emission-du-([\w-]+)\.html"
        match = re.search(regex, url)
        if match:
            return f'{match.group(1)}.mp4'
        else:
            return 'unknown-date.mp4'
    
    def get_video_urls(self):
        req = requests.get(self.source)
        soup = BeautifulSoup(req.text, 'html.parser')
        
        # Collect video URLs that match the pattern "emission-du"
        urls = {f'{self.pattern+a["href"]}' for a in soup.find_all('a', href=True) if "emission-du" in a['href']}
        return urls
    
    def get_video_urls_list(self):
        req = requests.get(self.source)
        soup = BeautifulSoup(req.text, 'html.parser')
        
        # Collect video URLs that match the pattern "emission-du"
        urls = [f'{self.pattern+a["href"]}' for a in soup.find_all('a', href=True) if "emission-du" in a['href']]
        return urls
    
    def download_last_video(self):
        urls = self.get_video_urls_list()
        if urls:
            last_url = urls[0]
            print(f"Found the last video: {last_url}. Downloading...")
            return self.download_video(last_url)
        else:
            print("No video found")
        return None

    def download_new_video(self):
        current_urls = set(self.get_video_urls())
        print("Current URLS: ",current_urls)
        # Find new URLs that have not been seen before
        new_urls = current_urls-self.urls
        new_video=None
        if new_urls:
            first_new_url = next(iter(new_urls))  # Get the first new URL
            print(f"Found a new video: {first_new_url}. Downloading...")
            new_video=self.download_video(first_new_url)
        else:
            print("No new video found")
        self.urls = current_urls
        return new_video

    def download_video(self,video_url):
        options = {
            'format': 'dash-video=5000000',  # Choose the appropriate format
            'outtmpl': download_folder + self.get_output_name(video_url)  # Save to your desired location
        }
        
        # Download the video using yt_dlp
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([video_url])
        return options['outtmpl']



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
        self.green_hsv = ((50,120,242),(80,210,255))
        self.rose_hsv=((140,230,223),(176,255,250))
        self.violet_hsv=((110,150,220),(131,170,255))
        self.blue_hsv=((90,100,100),(100,255,255))
        self.dark_blue_hsv = ((113,200,180),(115,255,255))
        if (mode==CYRIL):
            self.orange_hsv = ((15,200,200),(30,255,255))
            self.green_hsv = ((60,0,200),(70,255,255))
            self.blue_hsv = ((90,100,100),(100,255,255))
            self.dark_blue_hsv = ((113,200,180),(115,255,255))
        self.full_path=os.path.join( r'E:\Emissions\tlmvpsp',self.video_title)
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)
            print(f"Folder created: {self.full_path}")
        else:
            print(f"Folder already exists: {self.full_path}")
    
    def extract_question_from_frame_manche_1(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_id)
        ret, self.current_frame = self.cap.read()
        height,width,channels = self.current_frame.shape
        if (self.mode==CYRIL):
            cropped_img = self.current_frame[(80*height//100):(95*height//100),width//6:5*width//6,:]
        else: 
            cropped_img = self.current_frame[(85*height//100):(95*height//100),width//6:5*width//6,:]
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        if (self.mode==CYRIL):
            _, thresholded_img = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        else:
            _, thresholded_img = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
        question = pytesseract.image_to_string(thresholded_img, lang='fra')
        print(repr(question))
        self.questions.append(question.replace('\n',' '))

    def extract_question_from_frame_manche_2(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_id)
        ret, self.current_frame = self.cap.read()
        height,width,channels = self.current_frame.shape
        if (self.mode==CYRIL):
            cropped_img = self.current_frame[(80*height//100):(92*height//100),width//6:5*width//6,:]
        else:
            cropped_img = self.current_frame[(80*height//100):(89*height//100),width//6:5*width//6,:]
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        _, thresholded_img = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
        question = pytesseract.image_to_string(thresholded_img, lang='fra')
        print(repr(question))
        self.questions.append(question.replace('\n',' '))

    def extract_theme(self):
        height,width,channels = self.current_frame.shape
        if (self.mode==CYRIL):
            cropped_img = self.current_frame[(92*height//100):(96*height//100),36*width//100:64*width//100,:]
        else:
            cropped_img = self.current_frame[(90*height//100):(96*height//100),35*width//100:65*width//100,:]
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        if (self.mode==CYRIL):
            _, thresholded_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        else:
            _, thresholded_img = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        theme = pytesseract.image_to_string(thresholded_img, lang='fra')
        print(repr(theme))
        self.themes.append(theme.replace("\n",""))

    def analyze_color(self, frame, target_color_bgr, tolerance=10):
        height, width = frame.shape[:2]
        if (self.mode==CYRIL):
            cropped_frame = cv2.resize(frame[70*height//100:92*height//100, width//6:(width-width//6)], (width//4, height//4))
        cropped_frame = cv2.resize(frame[5*height//8:8*height//8, width//6:(width-width//6)], (width//4, height//4))
        cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

        # Define a range around the target color
        lower_bound,upper_bound=target_color_bgr

        # Create a mask for the target color
        mask = cv2.inRange(cropped_frame, lower_bound, upper_bound)

        # Calculate the ratio of target color pixels
        target_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        color_ratio = (target_pixels / total_pixels) * 100
        if DEBUG:
            print(self.current_frame_id,target_color_bgr,color_ratio)
        return color_ratio

    def look_for_color(self, color, threshold=0.001,up_threshold=100):
        green_up_threshold=6 if self.mode==CYRIL else 1
        while self.current_frame_id < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_id)
            ret, self.current_frame = self.cap.read()
            if ret:
                analyze_color = self.analyze_color(self.current_frame, color)
                if analyze_color > threshold and analyze_color<up_threshold and ( (color!=self.orange_hsv and color!=self.dark_blue_hsv and color!=self.blue_hsv) or self.analyze_color(self.current_frame,self.green_hsv)<green_up_threshold ):
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
        if (self.mode==CYRIL):
            threshold=18.45
        else:
            threshold=1
        self.look_for_color(self.orange_hsv,threshold)
        self.speed=self.frame_rate
        print("green")
        if (self.mode==CYRIL):
            threshold=6
            up_threshold=100
        else:
            threshold=1.375
            up_threshold=100
        self.look_for_color(self.green_hsv,threshold,up_threshold)
        self.speed=self.frame_rate*2
        self.extract_question_from_frame_manche_1()
    
    def find_all_questions_round_1_and_save(self):
        i=0
        if (self.mode!=NAGUI):
            self.find_one_question_round_1()
            filename=os.path.join(self.full_path,f'Question-Manche1-{i:03}.jpg')
            cv2.imwrite(filename,self.current_frame)
            i+=1
        for j in range(6):
            self.current_frame_id+=50*self.frame_rate
            self.find_one_question_round_1()
            filename=os.path.join(self.full_path,f'Question-Manche1-{i:03}.jpg')
            cv2.imwrite(filename,self.current_frame)
            i+=1
            self.find_one_question_round_1()
            filename=os.path.join(self.full_path,f'Question-Manche1-{i:03}.jpg')
            cv2.imwrite(filename,self.current_frame)
            i+=1
            if (self.mode==CYRIL and j==2):
                self.find_one_question_round_1()
                filename=os.path.join(self.full_path,f'Question-Manche1-{i:03}.jpg')
                cv2.imwrite(filename,self.current_frame)
                i+=1
        if (self.mode!=NAGUI):
            self.find_one_question_round_1()
            filename=os.path.join(self.full_path,f'Question-Manche1-{i:03}.jpg')
            cv2.imwrite(filename,self.current_frame)
    
    def find_one_question_round_2(self):
        if (self.mode!=NAGUI and self.mode!=CYRIL):
            print("rose")
            self.look_for_color(self.rose_hsv,1)
            self.speed=self.frame_rate//4
            print("green")
            self.look_for_color(self.green_hsv,2)
            self.speed=self.frame_rate*5
            self.extract_question_from_frame_manche_2()
        elif (self.mode==CYRIL):
            print("blue")
            self.look_for_color(self.blue_hsv,10)
            self.speed=self.frame_rate//4
            print("green")
            self.look_for_color(self.green_hsv,threshold=6)
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
            self.current_frame_id+=20*self.frame_rate
            filename=os.path.join(self.full_path,f'Question-Manche2-{i:03}.jpg')
            cv2.imwrite(filename,self.current_frame)
        self.extract_theme()
        

    
    def find_one_question_round_3_champion(self):
        if (self.mode!=NAGUI and self.mode!=CYRIL):
            print("rose")
            self.look_for_color(self.rose_hsv,2)
            self.speed=self.frame_rate//2
            print("green")
            self.look_for_color(self.green_hsv,2.01)
            self.speed=self.frame_rate*5
            self.extract_question_from_frame_manche_2()
        elif (self.mode==CYRIL):
            print("dark blue")
            self.look_for_color(self.dark_blue_hsv,5)
            self.speed=self.frame_rate//2
            print("green")
            self.look_for_color(self.green_hsv,6)
            self.speed=self.frame_rate*3
            self.extract_question_from_frame_manche_2()
    
    def find_one_question_round_3_challenger(self):
        if (self.mode!=NAGUI and self.mode!=CYRIL):
            print("violet")
            self.look_for_color(self.violet_hsv,1.8)
            self.speed=self.frame_rate//2
            print("green")
            self.look_for_color(self.green_hsv,1.88)
            self.speed=self.frame_rate*3
            self.extract_question_from_frame_manche_2()
        elif (self.mode==CYRIL):
            print("dark blue")
            self.look_for_color(self.dark_blue_hsv,5)
            self.speed=self.frame_rate//2
            print("green")
            self.look_for_color(self.green_hsv,6)
            self.speed=self.frame_rate*2
            self.extract_question_from_frame_manche_2()
    
    def find_all_questions_round_3_and_save(self):
        nb=0
        if (self.mode==JARRY):
            nb=5
        else:
            nb=6
        for i in range(nb):
            self.find_one_question_round_3_champion()
            self.current_frame_id+=4*self.frame_rate
            filename=os.path.join(self.full_path,f'Question-Manche3-{i:03}.jpg')
            cv2.imwrite(filename,self.current_frame)
        self.extract_theme()
        for i in range(nb):
            self.find_one_question_round_3_challenger()
            self.current_frame_id+=3*self.frame_rate
            filename=os.path.join(self.full_path,f'Question-Manche3-{6+i:03}.jpg')
            cv2.imwrite(filename,self.current_frame)
        self.extract_theme()
        
    
    def run(self):
        print("*"*50+"\nROUND 1\n"+"*"*50)
        self.current_frame_id=1*60*self.frame_rate
        self.find_all_questions_round_1_and_save()
        print("*"*50+"\nROUND 2\n"+"*"*50)
        self.current_frame_id+=2.5*60*self.frame_rate
        self.find_all_questions_round_2_and_save()
        print("*"*50+"\nROUND 3\n"+"*"*50)
        self.current_frame_id+=8*60*self.frame_rate
        self.find_all_questions_round_3_and_save()
        fullname=os.path.join(self.full_path,"questions.txt")
        with open(fullname, 'w') as file:
            for string in self.questions:
                if string=="":
                    string="Question non trouvée"
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
        elif (mode=="CYRIL"):
            mode=CYRIL
        else:
            print("CYRIL mode by default")
            mode=CYRIL
        if video_path=="auto":
            myVideoDownloader=VideoDownloader("https://www.france.tv/france-2/tout-le-monde-veut-prendre-sa-place/","https://www.france.tv")
            while True:
                new_video=myVideoDownloader.download_new_video()
                if new_video is not None:
                    video_path=new_video["default"]
                    question_detector = QuestionDetector(video_path,mode)
                    question_detector.run()
                    load_dotenv()
                    url = os.getenv("WEBHOOK_URL_2")
                    webhook.send_emission(url,question_detector.full_path)
                    break
                time.sleep(3600*2)
        elif video_path=="last":
            myVideoDownloader=VideoDownloader("https://www.france.tv/france-2/tout-le-monde-veut-prendre-sa-place/","https://www.france.tv")
            last_video=myVideoDownloader.download_last_video()
            video_path=last_video["default"]
        question_detector = QuestionDetector(video_path,mode)
        question_detector.run()
        load_dotenv()
        url = os.getenv("WEBHOOK_URL_2")
        webhook.send_emission(url,question_detector.full_path)
       
    else:
        print("Usage: python showtoquestions.py <video_path> <mode>")
        print("Example: python showtoquestions.py 'E:\Emissions\Tout le monde veut prendre sa place_France 2_2024_01_04_11_57.mp4' JARRY")