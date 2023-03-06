from collections import OrderedDict
import torch
from torchvision import models
from unet_parts import *
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def l2norm(X):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class TripletLoss(nn.Module):
    """
    Triplet loss: exemples positifs = les éléments avec id identique
    exemples négatifs = l' éléments d'id différent le plus similaires à l'anchor selon le modèle dans le batch.
    """

    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, im, ids):
        # exemples positifs
        positive = np.array(
            [np.where(np.array(ids) == id) for id in np.array(ids)]
        )  # shape (batch_size, vary)

        # exemples negatifs
        negative = np.array(
            [np.where(np.array(ids) != id) for id in np.array(ids)]
        )  # shape (batch_size, vary)

        # matrice de similarites entre les images encodees
        scores = im.mm(im.t())  # shape (batch_size, batch_size)

        cost = 0

        for j, pos in enumerate(positive):
            # get negatives for j-th anchor
            neg = negative[j]

            scores[j][j] = 0
            scores_anchor = scores[j]

            # compare maximum of similarity with negatives, with minimum similarity to positives to loss
            cost += max(
                self.margin
                + torch.max(scores_anchor[neg])
                - torch.min(scores_anchor[pos]),
                0,
            )
        return cost


class ReIdModel(nn.Module):
    def __init__(self, opt):
        """Load pretrained VGG16 or RESNET50 and replace top fc layer."""
        super(ReIdModel, self).__init__()
        self.embed_size = opt.embed_size

        # Load a pre-trained model
        self.cnn = self.get_cnn(opt)

        self.cnn_type = opt.cnn_type
        # Replace the last fully connected layer of CNN with a new one
        if opt.cnn_type.startswith("vgg"):
            self.fc = nn.Linear(self.cnn.classifier._modules["6"].in_features, 512)
            self.fc2 = nn.Linear(512, opt.embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1]
            )
        else:
            self.fc = nn.Linear(32768, 1024)
            self.fc2 = nn.Linear(1024, opt.embed_size)

        self.init_weights()
        self.criterion = TripletLoss(margin=opt.margin)
        params = list(self.fc.parameters())
        params += list(self.fc2.parameters())
        params += list(self.cnn.parameters())
        self.params = params
        self.Eiters = 0
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        if torch.cuda.is_available():
            self.cuda()

    def set_logger(self, logger):
        self.logger = logger

    def get_cnn(self, opt, train=False):
        if "vgg" in opt.cnn_type:
            model = models.__dict__[opt.cnn_type](pretrained=True)
            model.features = nn.DataParallel(model.features)
        elif "unet" in opt.cnn_type:
            model = UNet(opt)
            if train:
                model.load_state_dict(
                    opt.reidmodel_path, map_location=lambda storage, loc: storage
                )
        else:
            model = models.__dict__[opt.cnn_type](pretrained=True)
            # model = nn.DataParallel(model).cuda()
        return model

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if "vgg" in self.cnn_type:
            if "cnn.classifier.1.weight" in state_dict:
                state_dict["cnn.classifier.0.weight"] = state_dict[
                    "cnn.classifier.1.weight"
                ]
                del state_dict["cnn.classifier.1.weight"]
                state_dict["cnn.classifier.0.bias"] = state_dict[
                    "cnn.classifier.1.bias"
                ]
                del state_dict["cnn.classifier.1.bias"]
                state_dict["cnn.classifier.3.weight"] = state_dict[
                    "cnn.classifier.4.weight"
                ]
                del state_dict["cnn.classifier.4.weight"]
                state_dict["cnn.classifier.3.bias"] = state_dict[
                    "cnn.classifier.4.bias"
                ]
                del state_dict["cnn.classifier.4.bias"]
        elif "unet" in self.cnn_type:
            for key in state_dict.keys():
                # rename layers to be compative with the Extractor class
                if "cnn" in key:
                    key = key[4:]
        super(ReIdModel, self).load_state_dict(state_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.0) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images = images.to(device=device, dtype=torch.float32)
        image = images.permute(0, 3, 1, 2)
        features = self.cnn(images)
        # normalization in the image embedding space
        features = l2norm(features)

        features = self.fc(features)
        features = self.fc2(F.relu(features))
        features = l2norm(features)
        return features

    def forward_loss(self, img_emb, ids, **kwargs):
        """Compute the triplet loss given image embeddings"""
        loss = self.criterion(img_emb, ids)
        self.logger.update("Le", loss.item(), img_emb.size(0))
        self.loss_t = loss
        return loss

    def train_emb(self, images, indexes, ids, *args):
        """One training step."""
        self.Eiters += 1
        self.logger.update("lr", self.optimizer.param_groups[0]["lr"])
        self.logger.update("Eit", self.Eiters)

        # compute the embeddings
        img_emb = self.forward(images)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, ids)
        # print("training loss: ", loss)
        # compute gradient and do SGD step

        loss.backward()
        self.optimizer.step()


class UNet(nn.Module):
    def __init__(self, opt, bilinear=False):
        super(UNet, self).__init__()
        # self.n_channels = opt.n_channels
        # self.n_classes = opt.n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return self.flatten(x5)

    def load_state_dict(self, state_dict, map_location=torch.device("cpu")):

        new_state_dict = OrderedDict()

        state_dict_old = torch.load(
            state_dict, map_location=lambda storage, loc: storage
        )

        for key in state_dict_old.keys():
            # overlook weights of the upsampling side
            if ("up" in key) or ("outc" in key):
                ()

            # rename layers to be compative with the Extractor class
            elif "cnn" in key:
                new_state_dict[key[4:]] = state_dict_old[key]
            else:
                new_state_dict[key] = state_dict_old[key]
        super(UNet, self).load_state_dict(new_state_dict)
