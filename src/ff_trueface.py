import os
import torch
from torchvision import datasets, transforms

from PIL import Image
import numpy as np

class FF_TrueFace(datasets.DatasetFolder):
    def __init__(self, opt, phase="test",num_classes=2):
        self.opt = opt
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                            transforms.Resize([opt.input.image_size, opt.input.image_size], antialias=True)
                        ])
        # Whatsapp+StyleGAN3 for TrueFaceExtended;
        # Other folders contains real images, they're only needed to bilance the stylegan3 fake images
        # self.exclude = ['Whatsapp', "StyleGAN3", "09000", "10000", "11000", "12000", "13000"]
        # self.exclude = ['Whatsapp', "StyleGAN3"]
        self.samples = []

        if phase in ["train", "val"]:
            if opt.input.pre_post == "pre":
                dataset_path = os.path.join(opt.input.path, "Train/TrueFace_PreSocial")
            elif opt.input.pre_post == "post":
                dataset_path = os.path.join(opt.input.path, "Train/TrueFace_PostSocial")
        elif phase == "test":
            if opt.input.pre_post == "pre":
                dataset_path = os.path.join(opt.input.path, "Test/TrueFace_PreSocial")
            elif opt.input.pre_post == "post":
                dataset_path = os.path.join(opt.input.path, "Test/TrueFace_PostSocial")

        if opt.input.pre_post == "pre":
            self.exclude = []
        elif opt.input.pre_post == "post":
            self.exclude = ['Whatsapp', "StyleGAN3", "09000", "10000", "11000", "12000", "13000"]
        
        for root_1, dirs_1, files_1 in os.walk(dataset_path, topdown=True):
            for entry in sorted(dirs_1):
                data_folder = os.path.join(root_1, entry)
                if entry == 'FFHQ' or entry == 'Real' or entry == '0_Real':
                    for root, dirs, files in os.walk(data_folder, topdown=True):
                        if all([i not in root for i in self.exclude]):
                            for file in sorted(files):
                                if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")):
                                    item = os.path.join(root, file), torch.tensor(0)
                                    self.samples.append(item)
                elif (entry=='Fake' or entry=='1_Fake'):
                    for root, dirs, files in os.walk(data_folder, topdown=True):
                        if all([i not in root for i in self.exclude]):
                            for file in sorted(files):
                                if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")):
                                    item = os.path.join(root, file), torch.tensor(1)
                                    self.samples.append(item)

        print("Loaded " + str(len(self.samples)) + " images")


    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.samples)

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            class_label.clone().detach(), num_classes=self.num_classes
        )
        pos_sample = sample.clone()
        # pos_sample[:, 0, : self.num_classes] = one_hot_label
        
        pos_sample = pos_sample.reshape(1, -1)
        pos_sample = torch.concatenate((pos_sample, torch.unsqueeze(one_hot_label, 0)), -1)
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        # Create randomly sampled one-hot label.
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        )
        neg_sample = sample.clone()
        # neg_sample[:, 0, : self.num_classes] = one_hot_label

        neg_sample = neg_sample.reshape(1, -1)
        neg_sample = torch.concatenate((neg_sample, torch.unsqueeze(one_hot_label, 0)), -1)
        return neg_sample

    def _get_neutral_sample(self, z):
        # z[:, 0, : self.num_classes] = self.uniform_label

        z = z.reshape(1, -1)
        z = torch.concatenate((z, torch.unsqueeze(self.uniform_label, 0)), -1)
        return z

    def _generate_sample(self, index):
        # Get TrueFace sample.
        path, class_label = self.samples[index]
        sample = self.transform(Image.open(path))

        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        return pos_sample, neg_sample, neutral_sample, class_label