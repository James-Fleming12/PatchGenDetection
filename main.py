from train import trainJEPA, trainMOCO

def main():
    jepa_model = trainJEPA()
    moco_model = trainMOCO()

if __name__=="__main__":
    main()