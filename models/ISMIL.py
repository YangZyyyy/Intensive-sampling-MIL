import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

class WSI_dataset(Dataset):

    def __init__(self, wsi, coords, size, patch_size):
        self.patch_size = patch_size
        self.coords = coords
        self.wsi = wsi
        self.roi_transforms = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        img = self.wsi.read_region(coord, 0, (self.patch_size, self.patch_size)).convert('RGB')
        img = self.roi_transforms(img)
        return img


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class Branch(nn.Module):

    def __init__(self, features_size, n_classes, k_sample=3, subtyping=False):
        super(Branch, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(features_size, 512),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.attention_net = Attn_Net_Gated(512, 256, dropout=True)

        self.classifier = nn.Linear(512, n_classes)
        self.instance_classifiers = nn.ModuleList([nn.Linear(512, 2) for i in range(n_classes)])
        self.k_sample = k_sample
        self.instance_loss_fn = nn.CrossEntropyLoss()
        self.n_classes = n_classes
        self.subtyping = subtyping

    def forward(self, x, label=None, inst_inference=True):
        device = x.device

        # attention based MIL
        h = self.mlp(x)
        A, h = self.attention_net(h) # A: n * 1 h: n * 512
        A_soft = torch.softmax(A, dim=0)
        M = torch.mm(torch.transpose(A_soft, 1, 0), h) # 1 * 512
        logits = self.classifier(M)

        res = {
            'logits': logits,
            'attention_raw': A,
            'M': M
        }

        if inst_inference:
            # instance classify
            inst_logits = self.classifier(h)
            res['inst_logits'] = inst_logits


        # CLAM_Branch
        if label is not None:
            total_inst_loss = 0.0
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:
                    instance_loss = self.inst_eval(A, h, classifier)
                else:
                    continue

                total_inst_loss += instance_loss
            res['inst_loss'] = total_inst_loss

        return res

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def inst_eval(self, A, h, classifier):
        device = h.device
        A = A.squeeze()

        if len(h) < 10 * self.k_sample:
            k = 1
        else:
            k = self.k_sample
        top_p_ids = torch.topk(A, k)[1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k)[1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(k, device)
        n_targets = self.create_negative_targets(k, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss


class ISMIL(nn.Module):

    def __init__(self, features_size, n_classes, topk=3, threshold=0.5, neighk=24):
        super(ISMIL, self).__init__()
        self.topk=topk
        self.threshold = threshold
        self.neighk = neighk

        self.branch_1 = Branch(features_size, n_classes)
        self.branch_2 = Branch(features_size, n_classes)
        self.classifier = nn.Linear(1024, n_classes)


    def compute_inst_probs(self, inst_logits, attention_raw):
        return torch.sigmoid(attention_raw.squeeze()) * torch.softmax(inst_logits, dim=-1)[:, 1]

    def compute_coords(self, inst_probs, coords):
        topk_index = torch.topk(inst_probs, k=min(self.topk, len(inst_probs)))[1]
        over_threshold_index = torch.where(inst_probs > self.threshold)[0]
        index = torch.unique(torch.cat([topk_index, over_threshold_index]))
        return coords[index.cpu().numpy()]

    def forward(self, x1, x2, coords1, coords2, label=None):
        res_1 = self.branch_1(x1, label)

        inst_logits, attention_raw = res_1['inst_logits'], res_1['attention_raw']
        inst_probs = self.compute_inst_probs(inst_logits, attention_raw)
        roi_coords = self.compute_coords(inst_probs, coords1)

        neigh = NearestNeighbors(n_neighbors=min(24, len(x2)))
        neigh.fit(coords2.cpu().numpy())
        query_index = neigh.kneighbors(roi_coords.cpu().numpy())[1]
        query_index = np.unique(query_index.flatten())

        x2 = x2[query_index]
        res_2 = self.branch_2(x2, inst_inference=False)

        M = torch.cat([res_1['M'], res_2['M']], dim=-1)
        logits = self.classifier(M)

        res = {
            'logits_3': logits,
            'logits_1': res_1['logits'],
            'logits_2': res_2['logits']
        }

        if label is not None:
            res['inst_loss'] = res_1['inst_loss']


        return res

    def patch_probs(self, x, coords, wsi, patch_size, feature_extractor, sample_num=3, **kwargs):
        with torch.no_grad():
            res_1 = self.branch_1(x, inst_inference=True)
            A = res_1['attention_raw']
            inst_logits = res_1['inst_logits']
            inst_probs = self.compute_inst_probs(inst_logits, A)
            roi_coords = self.compute_coords(inst_probs, coords)


            mark = {}
            sampled_coords = []
            for coord in roi_coords:
                coord_x, coord_y = coord
                for a in range(coord_x-patch_size//2, coord_x + patch_size//2, patch_size // sample_num):
                    if a not in mark:
                        mark[a] = set()
                    for b in range(coord_y-patch_size//2, coord_y + patch_size//2, patch_size // sample_num):
                        if b not in mark[a]:
                            sampled_coords.append([a, b])
                            mark[a].add(b)

            sampled_coords = np.array(sampled_coords)
            sampled_set = WSI_dataset(wsi, sampled_coords, 224, patch_size)
            dataloader = DataLoader(sampled_set, batch_size=256)

            rs_features = []
            for batch in tqdm(dataloader, 'resample'):
                batch = batch.to(x.device)
                rs_features.append(feature_extractor(batch))
            rs_features = torch.cat(rs_features, dim=0)

            res_2 = self.branch_2(rs_features, inst_inference=True)
            rs_inst_probs = self.compute_inst_probs(res_2['inst_logits'], res_2['attention_raw'])

            inst_probs = torch.cat([inst_probs, rs_inst_probs])
            coords = np.concatenate([coords, sampled_coords], axis=0)

            M = torch.cat([res_1['M'], res_2['M']], dim=-1)
            logits = self.classifier(M)
            probs = torch.softmax(logits, dim=-1)

            return {
                'probs': probs,
                'inst_probs': inst_probs,
                'coords': coords
            }
