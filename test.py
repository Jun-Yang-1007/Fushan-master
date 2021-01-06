import os
import time

def main():
    print (" main init ")
def createFile():

    print (os.getcwd())

    ttsv = time.strftime("%Y%m%d%H%M%S") + ".txt"
    ##tts = time.strptime(ttsv,"%Y-%m-%d %H:%M:%S")

    print (ttsv)

    os.chdir(r'E:\MyFpi\Project1\fujian-water-master-PY3\Data')
    os.mkdir(ttsv)


if __name__ == "__main__":

    print ("start")

    createFile()

    print ("end")

    os._exit(0)