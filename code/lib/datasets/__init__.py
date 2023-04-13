from .threedpw import ThreeDPWDataset, ThreeDPWValDataset, ThreeDPWTestDataset
from torch.utils.data import DataLoader

def find_dataset_using_name(name):
    mapping = {
        "ThreeDPW": ThreeDPWDataset,
        "ThreeDPWVal": ThreeDPWValDataset,
        "ThreeDPWTest": ThreeDPWTestDataset,
        "Hi4D": ThreeDPWDataset,
        "Hi4DVal": ThreeDPWValDataset,
        "Hi4DTest": ThreeDPWTestDataset,
    }
    cls = mapping.get(name, None)
    if cls is None:
        raise ValueError(f"Fail to find dataset {name}") 
    return cls


def create_dataset(opt):
    dataset_cls = find_dataset_using_name(opt.dataset)
    dataset = dataset_cls(opt)
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        drop_last=opt.drop_last,
        shuffle=opt.shuffle,
        num_workers=opt.worker,
        pin_memory=True
    )