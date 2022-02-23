import json
import gdown
from os import listdir
from os.path import isfile, join
from zipfile import ZipFile


g_drive_url = "https://drive.google.com/uc?id=1neRj5GnePbCE9vRaQYgDYLMoAdGVlWca"

def get_all_zero_data():
    paths = [f for f in listdir('./dataset') if isfile(join('./dataset', f)) and f.endswith('.json')]
    data = []
    for path in paths:
        data += json.load(open(path))
    return data

def download_data():
    gdown.download(g_drive_url, "./dataset/data.zip", quiet=False)
    with ZipFile("./dataset/data.zip", "r") as zip:
        zip.extractall('./dataset')
        zip.close()


