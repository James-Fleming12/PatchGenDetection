from train import trainJEPA, trainMOCO, DalleDataset, validateModel
from model import ConGenDetect, JepaGenDetect

from torch.utils.data import random_split, DataLoader

def main():
    dataset = DalleDataset(root_dir='data/dalle-recognition-dataset')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 200
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    jepa_model: JepaGenDetect = trainJEPA(train_loader)
    moco_model: ConGenDetect = trainMOCO(train_loader)

    jepa_val = validateModel(jepa_model, val_loader)
    moco_val = validateModel(moco_model, val_loader)

if __name__=="__main__":
    main()