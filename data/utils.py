from data.datasets import imagenet, oxford_pets, eurosat,\
    stanford_cars, caltech101, sun397, fgvc, food101, oxford_flowers, dtd, ucf101
from torch.utils.data import DataLoader


def set_loader(id_dataset, transforms=None, partition="test", batch_size=128, shots="all", num_workers=4):

    # Initiate dataset
    if id_dataset == "imagenet":
        dataset = imagenet.Dataset(transform=transforms, partition=partition, shots=shots)
    elif id_dataset == "oxfordPets":
        dataset = oxford_pets.Dataset(transform=transforms, partition=partition, shots=shots)
    elif id_dataset == "eurosat":
        dataset = eurosat.Dataset(transform=transforms, partition=partition, shots=shots)
    elif id_dataset == "stanfordCars":
        dataset = stanford_cars.Dataset(transform=transforms, partition=partition, shots=shots)
    elif id_dataset == "caltech":
        dataset = caltech101.Dataset(transform=transforms, partition=partition, shots=shots)
    elif id_dataset == "sun397":
        dataset = sun397.Dataset(transform=transforms, partition=partition, shots=shots)
    elif id_dataset == "aircraft":
        dataset = fgvc.Dataset(transform=transforms, partition=partition, shots=shots)
    elif id_dataset == "food101":
        dataset = food101.Dataset(transform=transforms, partition=partition, shots=shots)
    elif id_dataset == "flowers":
        dataset = oxford_flowers.Dataset(transform=transforms, partition=partition, shots=shots)
    elif id_dataset == "dtd":
        dataset = dtd.Dataset(transform=transforms, partition=partition, shots=shots)
    elif id_dataset == "ucf":
        dataset = ucf101.Dataset(transform=transforms, partition=partition, shots=shots)
    else:
        print("Dataset not supported")
        return None

    # Set dataloader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return loader