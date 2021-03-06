from torch.utils.data import Dataset, DataLoader


class LicenseDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, index):
        # augment_sample --> apply albumentation
        # labels2output_map --> reimplement
        return
