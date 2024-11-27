from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset


@dataclass
class BatchBanditFeedbackDataset(Dataset):
    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray

    def __post_init__(self):
        assert self.context.shape[0] == self.action.shape[0] == self.reward.shape[0] == self.pscore.shape[0]

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
        )

    def __len__(self):
        return self.context.shape[0]
