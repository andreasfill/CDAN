import argparse

from utils.parser import create_model, define_dataloader, define_network, define_dataset, parse
from utils.reproducibility import set_seed_and_cudnn
from torch.utils.data import Dataset, DataLoader
from typing import Iterable, Tuple
from torch import Tensor, load
from torchvision import transforms
from os.path import join
from os import listdir, getcwd
from PIL import Image
from einops import rearrange
import matplotlib.pyplot as plt
from models.cdan import CDAN
from models.model import Model
import cv2 as cv

class DLRDataset(Dataset):
    def __init__(
        self,
        strip: str,
        create_patches: bool):
        super().__init__()
        self.image_height: int = 6144 
        self.image_width: int = 8192
        # self.patch_size: int = 16
        # self.patch_size: int = 512
        self.patch_size: int = 2048
        self.num_horizontal_patches: int = self.image_width // self.patch_size
        self.num_vertical_patches: int = self.image_height // self.patch_size
        self.total_num_patches: int = self.num_horizontal_patches * self.num_vertical_patches
        print(f"Height: {self.image_height}, Width: {self.image_width}, Patch Size: {self.patch_size}, Vertical Patches: {self.num_vertical_patches}, Horizontal Patches: {self.num_horizontal_patches}, Total number of Patches: {self.total_num_patches}")
        self.create_patches: bool = create_patches
        
        low_light_root = join(getcwd(), "images", strip)
        # self.low_light_dataset = [join(low_light_root, image) for image in listdir(low_light_root)]
        # self.low_light_image_names = [image for image in listdir(low_light_root)]
        self.low_light_dataset = list()
        self.low_light_image_names = list()
        for image in listdir(low_light_root):
            self.low_light_image_names.append(image)
            self.low_light_dataset.append(join(low_light_root, image))
        # Original size is 8416x6032. Resize to closest multiple of 512, which is the image size
        # the model was trained on.
        self.transforms = transforms.Compose([
            # Resizing only works on PIL images and not on numpy arrays.
            # transforms.Resize((self.image_height, self.image_width)),
            # PIL reads image in width, height order.
            # transforms.Resize((self.image_width, self.image_height)),
            # Converts CWH to CHW and normalizes the image from [0..255] to [0..1] range.
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx) -> Tuple[str, Tensor]:
        image_name: str = self.low_light_image_names[idx]

        # low_light = Image.open(self.low_light_dataset[idx]).convert('RGB')
        low_light = cv.imread(self.low_light_dataset[idx], cv.IMREAD_COLOR_RGB)
        # Why does opencv load the image as HWC, but resizing takes the format WH???
        low_light = cv.resize(low_light, (self.image_width, self.image_height))

        # TODO: Do the same for the astronaut or other pictures and check if their RGB channels
        # are very different before and after enhancement.
        fig, sp = plt.subplots(nrows=1, ncols=3, figsize=((30, 10)), layout="constrained")
        for i, color_channel in enumerate(["Red", "Green", "Blue"]):
            color_bar = sp[i].imshow(low_light[:, :, i], cmap="viridis", vmin=0, vmax=255)
            sp[i].set_title(f"{color_channel} Color Channel")
            sp[i].axis("off")
            fig.colorbar(color_bar, ax=sp[i], shrink=0.65, label="Pixel Intensity")

        save_dir = join(getcwd(), "images", "strip1", "output")
        plt.savefig(join(save_dir, f"{self.patch_size}_OriginalColorChannels_{image_name}"))

        low_light = self.transforms(low_light)

        if self.create_patches:
            low_light = rearrange(low_light, "c (b1 h) (b2 w) -> (b1 b2) c h w", b1=self.num_vertical_patches, b2=self.num_horizontal_patches, h=self.patch_size, w=self.patch_size)

        return image_name, low_light

    def __len__(self) -> int:
        return len(self.low_light_dataset)

def main():
    set_seed_and_cudnn()

    # phase = config['phase']
    strip: str = "strip1"
    dataset: DLRDataset = DLRDataset(strip=strip, create_patches=True)
    dataloader: Iterable[Tensor] = DataLoader(dataset, batch_size=1, shuffle=False)
    network: CDAN = CDAN()
    network.load_state_dict(load("CDAN_weights.pt", map_location="cpu", weights_only=True))
    model: Model = Model(
        network=network,
        train_phase=False,
        device="cuda",
        is_dataset_paired=False,
        dataloader=dataloader,
        apply_post_processing=True,
        output_images_path=join(getcwd(), "images", strip, "output"),
        epoch=80,
        learning_rate=1e-3,
        dataset=dataset
    )
    # print(model.__dict__)
    
    model.test()
    

    # for low in dataloader:
    #     low = low.squeeze(dim=0)
    #     print(low.shape)
    #     fig, sp = plt.subplots(nrows=2, ncols=12, layout="constrained", figsize=((120, 20)))
    #     for i, img in enumerate(low[:24]):
    #         sp[i // 12, i % 12].imshow(img.numpy().transpose((1, 2, 0)))
    #         sp[i // 12, i % 12].axis("off")

        

    #     break

    # plt.savefig("image_as_patches.jpg")
        
    # dataset = define_dataset(config[phase]['dataset'])
    # dataloader = define_dataloader(dataset, config[phase]['dataloader']['args'])
    # network = define_network(config['model']['networks'][0])

    # model = create_model(config=config,
    #                      network=network,
    #                      dataloader=dataloader
    #                     )

    # if phase == 'train':
    #     model.train()
    # else:
    #     model.test()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', type=str, default='config/default.json', help='Path to the JSON configuration file')
    # parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], help='Phase to run (train or test)', default='train')

    # # parser configs
    # args = parser.parse_args()
    # config = parse(args)

    # main(config)
    main()