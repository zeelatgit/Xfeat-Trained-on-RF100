import json
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class RF100UnderwaterDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of the RF100 underwater dataset.
            split (str): One of 'train', 'valid', 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or transforms.ToTensor()

        # Paths
        self.images_dir = os.path.join(root_dir, split)
        self.annotations_path = os.path.join(root_dir, split, '_annotations.coco.json')

        # Load annotations (if needed for metadata)
        self.annotations = None
        if os.path.exists(self.annotations_path):
            with open(self.annotations_path, 'r') as f:
                coco_data = json.load(f)
            self.images = {img['id']: img for img in coco_data['images']}
            self.annotations = coco_data['annotations'] if 'annotations' in coco_data else None
        else:
            self.images = []

        # Create a list of image filenames
        self.image_files = [img['file_name'] for img in self.images.values()]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image
