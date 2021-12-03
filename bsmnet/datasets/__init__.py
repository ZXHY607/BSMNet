from .kitti_dataset import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .Drivingstereo_dataset import DrivingstereoDatset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "drivingstereo": DrivingstereoDatset
}
