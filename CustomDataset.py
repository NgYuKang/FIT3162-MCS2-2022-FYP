from torchvision.datasets import ImageFolder
class CustomDataset(ImageFolder):
    def __getitem__(self, index: int):
        path, label = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        target = {"main_label": label,
                 "sub_label": self.sub_labels[index]}
        return sample, target