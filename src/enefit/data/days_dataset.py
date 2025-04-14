import torch
from torch.utils.data import Dataset


class DaysDataset(Dataset):
    def __init__(self, data, lags, seq_len: int, save:bool = False):
        super().__init__()
        
        self.lags = lags

        self.seq_len = seq_len
        
        index = data.keys().to_list()
        index.remove('prediction_unit_id')
        
        self.means = [0.1277779782376821, 0.26107474891445476, 0.07862313143698685, 0.24906541229289986, 0.15429031000507026, 0.26429071124835923, 0.09041434149383237, 0.6492504504013048, 0.11064407989990507, 0.45457957923078396, 0.08409352756308007, 0.4617043748429825, 0.11872232959259765, 0.628484155104194]
        self.stds = [0.19453651424387805, 0.10260584476563492, 0.134744362977315, 0.09993685767144098, 0.22255086340003832, 0.09620643916073567, 0.16159597866935038, 0.23916194723685066, 0.1845046375508081, 0.17503611656780277, 0.1486883772037978, 0.25257191170364235, 0.18246006143611357, 0.2447319686556321]
        
        data.is_business = data.is_business.astype(int)
        
        groups = data.groupby('prediction_unit_id')
        
        # TODO: tune the lower and upper stride
        self.dataset = torch.stack([torch.tensor(group[index].iloc[: seq_len].to_numpy(), dtype=torch.float32) for _, group in groups], dim = 0)
        
        del groups

        
    def getFeature(self):
        return self.dataset.shape[2] -3  # columns to ignore

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        self.scaling = torch.tensor([(self.dataset[index, 0, 2] + 100)**0.5, (self.dataset[index, 0, 2] + 100)**0.5]) 
        self.is_business = int(self.dataset[index, 0, 0].item())
        self.product_type = int(self.dataset[index, 0, 1].item())

        self.mean  = torch.tensor([self.means[2 * (4 * self.is_business + self.product_type - 1)], self.means[2 * (4 * self.is_business + self.product_type - 1) + 1]])
        self.std = torch.tensor([self.stds[2 * (4 * self.is_business + self.product_type - 1)], self.stds[2 * (4 * self.is_business + self.product_type - 1) + 1]])

        input = self.dataset[index, :, 3:].clone()

        for i in range(len(self.lags) + 1):
            j = 2*i + 6
            input[:, j:j+2] = torch.div(torch.sub(torch.div(input[:, j:j+2] ** 0.5, self.scaling.unsqueeze(0)), self.mean.unsqueeze(0)), self.std.unsqueeze(0))

        return input, (self.mean , self.std, self.scaling)
