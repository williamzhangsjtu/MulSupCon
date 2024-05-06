from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
import torch
import os

class Sampler(object):
    """
    Base class of sampler
    """
    def __init__(
        self, 
        h5: str, 
        indices: list, 
        batch_size: int, 
        seed:int = 0, 
    ):
        """
        Parameters:
            h5 (str): path to the h5 file, containing numpy-array like data
            indices (list): indices of the data, indexing the data in the h5 file
            batch_size (int): batch size
            seed (int): random seed used to initialize the random state
        """

        self.h5, self.indices, self.batch_size,\
            self.seed = h5, indices, batch_size, seed
        self.random_state = np.random.RandomState(seed=seed)
        with h5py.File(h5, 'r') as input_data:
            targets = input_data['target'][:]
            # targets: B x n_class (one hot)
            self.n_class = targets.shape[-1]
        self.indices = np.array(self.indices)
        self.targets = targets[self.indices, :]

class RandomSampler(Sampler):
    def __init__(
        self, 
        h5: str, 
        indices: list, 
        batch_size: int, 
        seed:int = 0
    ):

        super(RandomSampler, self).__init__(
            h5=h5,
            indices=indices,
            batch_size=batch_size,
            seed=seed
        )
        self.ptr = 0
        self.random_state.shuffle(self.indices)
    
    def __iter__(self):
        # shuffle when all data has been iterated once
        while True:
            batch_indices = []
            i = 0
            while i < self.batch_size:
                index = self.indices[self.ptr]
                self.ptr = (self.ptr + 1) % len(self.indices)
                if not self.ptr:
                    self.random_state.shuffle(self.indices)
                batch_indices.append(index)
                i += 1
            yield batch_indices



class TrainImageDataset(Dataset):
    """
    Train Dataset for image data
    """
    def __init__(
            self,
            h5: str,
            transform: callable = lambda x: x
        ):
        """
        Parameters:
            h5 (str): path to the h5 file, containing numpy-array like data
            transform (callable): transformation function (e.g., data augmentation)
        """

        self.h5 = h5
        self.transform = transform

        with h5py.File(self.h5, 'r') as input:
            self.targets = input['target'][:]
            # UNCOMMENT the following lines if you want to load the data into memory
            # self.images = input['Image'][:]
        self.targets = torch.from_numpy(self.targets).to(torch.float32)
    def __getitem__(self, i):
        # COMMENT the following lines if you want to load the data into memory
        with h5py.File(self.h5, 'r') as input_h5:
            image = input_h5['Image'][i]

        # UNCOMMENT the following lines if you want to load the data into memory
        # image = self.images[i]
        
        target = self.targets[i]
        view1 = self.transform(image)
        view2 = self.transform(image)
        return torch.stack([view1, view2], dim=0), target
        
class EvalImageDataset(Dataset):
    """
    Evaluation Dataset for image data
    """
    def __init__(
            self,
            h5: str,
            indices: list,
            transform: callable = lambda x: x
        ):
        """
        Parameters:
            h5 (str): path to the h5 file, containing numpy-array like data
            indices (list): indices of the data, indexing the data in the h5 file
            transform (callable): transformation function (e.g., data augmentation)
        """

        self.h5 = h5
        self.indices = indices
        self.transform = transform
        with h5py.File(self.h5, 'r') as input:
            self.targets = input['target'][:]
            # UNCOMMENT the following lines if you want to load the data into memory
            # self.images = input['Image'][:]
        self.targets = torch.from_numpy(self.targets).to(torch.float32)
    def __getitem__(self, i):
        index = self.indices[i]

        # COMMENT the following lines if you want to load the data into memory
        with h5py.File(self.h5, 'r') as input_h5:
            image = input_h5['Image'][index]

        # UNCOMMENT the following lines if you want to load the data into memory
        # image = self.images[index]

        target = self.targets[index]
        view1 = self.transform(image)
        view2 = self.transform(image)
        return torch.stack([view1, view2], dim=0), target
    
    def __len__(self):
        return len(self.indices)


class TrainVectorDataset(Dataset):
    """
    Train Dataset for vector data
    """
    def __init__(
            self,
            h5: str,
            transform: callable = lambda x: x
        ):
        self.h5 = h5
        self.transform = transform
        with h5py.File(self.h5, 'r') as input:
            self.targets = input['target'][:]
            self.feature = input['feature'][:].astype(np.float32)
        self.targets = torch.from_numpy(self.targets).to(torch.float32)
    def __getitem__(self, i):
        # with h5py.File(self.h5, 'r') as input_h5:
        #     feature = input_h5['feature'][i]
        feature = self.feature[i]
        
        target = self.targets[i]
        view1 = self.transform(feature.reshape(1, -1)).reshape(-1)
        view2 = self.transform(feature.reshape(1, -1)).reshape(-1)
        return torch.stack([view1, view2], dim=0), target
        
class EvalVectorDataset(Dataset):
    """
    Evaluation Dataset for vector data
    """
    def __init__(self,
                 h5: str,
                 indices: list,
                 transform: callable = lambda x: x):
        self.h5 = h5
        self.indices = indices
        self.transform = transform
        with h5py.File(self.h5, 'r') as input:
            self.targets = input['target'][:]
            self.feature = input['feature'][:].astype(np.float32)
        self.targets = torch.from_numpy(self.targets).to(torch.float32)
    def __getitem__(self, i):
        index = self.indices[i]

        # with h5py.File(self.h5, 'r') as input_h5:
        #     feature = input_h5['feature'][index]
        feature = self.feature[index]

        target = self.targets[index]
        view1 = self.transform(feature.reshape(1, -1)).reshape(-1)
        view2 = self.transform(feature.reshape(1, -1)).reshape(-1)
        return torch.stack([view1, view2], dim=0), target
    
    def __len__(self):
        return len(self.indices)

def create_dataloader(
    input_h5: str,
    indices: list,
    is_eval: bool = False,
    seed: int = 0,
    transform: callable = lambda x: x,
    is_image: bool = True,
    **kwargs
):
    """
    Parameters:
        input_h5 (str): path to the h5 file, containing numpy-array like data
        indices (list): indices of the data, indexing the data in the h5 file
        is_eval (bool): whether the dataloader is for evaluation
        seed (int): random seed used to initialize the random state
        transform (callable): transformation function (e.g., data augmentation)
        is_image (bool): whether the data is image data
    """

    batch_size = kwargs.setdefault('batch_size', 64)
    num_workers = kwargs.setdefault('num_workers', 16)

    if is_eval:
        if is_image:
            dataset = EvalImageDataset(
                h5=input_h5,
                indices=indices,
                transform=transform
            )
        else:
            dataset = EvalVectorDataset(
                h5=input_h5,
                indices=indices,
                transform=transform
            )
        return DataLoader(dataset, shuffle=False, **kwargs)
    if is_image:
        dataset = TrainImageDataset(input_h5, transform)
    else:
        dataset = TrainVectorDataset(input_h5, transform)

    sampler = RandomSampler(
        h5=input_h5,
        indices=indices,
        batch_size=batch_size,
        seed=seed
    )
    return DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=num_workers
    )







class RandomSampler2(Sampler):
    def __init__(
        self, 
        h5: str, 
        indices: list, 
        batch_size: int, 
        seed:int = 0
    ):

        super(RandomSampler2, self).__init__(
            h5=h5,
            indices=indices,
            batch_size=batch_size,
            seed=seed
        )
        self.ptr = 0
        self.indices = np.array(range(len(indices)))
        self.random_state.shuffle(self.indices)
    
    def __iter__(self):

        while True:
            batch_indices = []
            i = 0
            while i < self.batch_size:
                index = self.indices[self.ptr]
                self.ptr = (self.ptr + 1) % len(self.indices)
                if not self.ptr:
                    self.random_state.shuffle(self.indices)
                batch_indices.append(index)
                i += 1
            yield batch_indices

    
class TrainImageDataset2(Dataset):
    def __init__(self, h5, indices, transform=lambda x: x):
        self.h5 = h5
        self.transform = transform
        self.indices = sorted(indices, reverse=False)
        with h5py.File(self.h5, 'r') as input:
            self.targets = input['target'][:]
            self.images = input['Image'][self.indices]
        self.targets = torch.from_numpy(self.targets).to(torch.float32)
    def __getitem__(self, i):
        image = self.images[i]
        
        target = self.targets[i]
        view1 = self.transform(image)
        view2 = self.transform(image)
        return torch.stack([view1, view2], dim=0), target
        
class EvalImageDataset2(Dataset):
    def __init__(self, h5, indices, transform=lambda x: x):
        self.h5 = h5
        self.indices = sorted(indices, reverse=False)
        self.transform = transform
        with h5py.File(self.h5, 'r') as input:
            self.targets = input['target'][:]
            self.images = input['Image'][self.indices]
        self.targets = torch.from_numpy(self.targets).to(torch.float32)
    def __getitem__(self, i):
        image = self.images[i]
        target = self.targets[i]
        view1 = self.transform(image)
        view2 = self.transform(image)
        return torch.stack([view1, view2], dim=0), target
    
    def __len__(self):
        return len(self.indices)