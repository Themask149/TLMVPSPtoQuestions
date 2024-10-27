import requests
from discord import SyncWebhook, File
import discord
import os
from dotenv import load_dotenv
import sys

CYRIL_bp=(15,27,33)
CYRIL=4
mode=CYRIL

def send_emission(url, directory_path):
    wh = SyncWebhook.from_url(url)
    questions=[]
    themes=[]
    wh.send(content="C'est parti pour le Cyril de ce midi ! Voici les questions :")
    with open(os.path.join(directory_path,"questions.txt"), 'r') as file:
        for line in file:
            questions.append(line.strip())

    with open(os.path.join(directory_path,"themes.txt"), 'r') as file:
        for line in file:
            themes.append(line.strip())
    if mode==CYRIL:
        bp1,bp2,bp3=CYRIL_bp
    for i,filename in enumerate(sorted(os.listdir(directory_path))):
        file_path = os.path.join(directory_path,filename)
        if filename=="questions.txt" or filename=="themes.txt":
            break
        if os.path.isfile(file_path):
            if i==bp1:
                wh.send(content=f"Le Thème de la manche 2 est : {themes[0]}")
            elif i==bp2:
                wh.send(content=f"Le Thème du champion est : {themes[1]}")
            elif i==bp3:
                wh.send(content=f"Le Thème du challenger est : {themes[2]}")
            wh.send(content=questions[i])
            file = discord.File(file_path, filename="SPOILER_"+filename)
            wh.send(file=file)
            print(f"{filename} is sent")


if __name__ == "__main__":
    if len(sys.argv)!=2:
        print("Usage: python webhook.py directory_path")
        sys.exit(1)
    directory_path = sys.argv[1]
    load_dotenv()
    url = os.getenv("WEBHOOK_URL_2")
    send_emission(url,directory_path)