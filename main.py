from train import trainJEPA, trainMOCO
from model import ConGenDetect, JepaGenDetect

def main():
    jepa_model: JepaGenDetect = trainJEPA()
    moco_model: ConGenDetect = trainMOCO()

if __name__=="__main__":
    main()