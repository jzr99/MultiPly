from .dtu import DTUDataset, DTUValDataset
from .sample import SampleDataset, SampleValDataset, SampleTestDataset
from .cape import CapeDataset, CapeValDataset, CapeTestDataset
from .poseprior import PosePriorDataset, PosePriorValDataset
from .real import RealDataset, RealValDataset
from .buff_mono import BuffMonoDataset, BuffMonoValDataset, BuffMonoTestDataset
from .buff_mono_seg import BuffMonoSegDataset, BuffMonoSegValDataset, BuffMonoSegTestDataset
from .threedpw import ThreeDPWDataset, ThreeDPWValDataset
from torch.utils.data import DataLoader

def find_dataset_using_name(name):
    mapping = {
        "DTU": DTUDataset,
        "DTUVal": DTUValDataset,
        "Sample": SampleDataset,
        "SampleVal": SampleValDataset,
        "SampleTest": SampleTestDataset,
        "Cape": CapeDataset,
        "CapeVal": CapeValDataset,
        "CapeTest": CapeTestDataset,
        "PosePrior": PosePriorDataset,
        "PosePriorVal": PosePriorValDataset,
        "Real": RealDataset,
        "RealVal": RealValDataset,
        "BuffMono": BuffMonoDataset,
        "BuffMonoVal": BuffMonoValDataset,
        "BuffMonoTest": BuffMonoTestDataset,
        "BuffMonoSeg": BuffMonoSegDataset,
        "BuffMonoSegVal": BuffMonoSegValDataset,
        "BuffMonoSegTest": BuffMonoSegTestDataset,
        "ThreeDPW": ThreeDPWDataset,
        "ThreeDPWVal": ThreeDPWValDataset
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