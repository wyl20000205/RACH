import argparse
import logging
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from optimization import DPDH_Adam
from config import cfg

# from dataloader import dataloader
from triplet.losses import TripletCustomMarginLoss, LowerBoundLoss, bit_var_loss
from triplet.methods import MetricLearningMethods
from triplet.miners.triplet_automargin_miner import TripletAutoParamsMiner
from pytorch_metric_learning import distances, reducers
from model_daph import DPDH_Encoder
from loss import (
    DPDH_LOSS,
    DNPH_LOSS,
    CPF,
    quantization_Loss,
    multilabelsimilarity_loss,
    noise_loss,
    our_loss,
    Cross_modal_class_balance_loss,
    DSPH,
)
from rot import Rot
from metrics import (
    calc_map_k_matrix,
    pr_curve,
    p_top,
    compute_ndcg_at_n,
    calc_precisions_hamming_radius,
)
from load_data import generate_dataset

logging.basicConfig(
    filename="./log.txt", level=logging.INFO, format="%(asctime)s - %(message)s"
)


def get_config():
    parser = argparse.ArgumentParser(description="AdaTriplet")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--method", type=str, default="AdaTriplet-AM")
    parser.add_argument("--type_of_triplets", type=str, default="semihard")
    parser.add_argument("--dataset", type=str, default="mirflickr25k")
    parser.add_argument("--bit", type=int, default=64)

    return parser.parse_args()


class WYL_Trainer:
    def __init__(self):

        self.model = DPDH_Encoder().float().to(cfg["device"])
        self.dpdh = DPDH_LOSS().to(cfg["device"])
        self.cpf = CPF().to(cfg["device"])
        self.dnph = DNPH_LOSS().to(cfg["device"])
        self.dsph = DSPH().to(cfg["device"])
        self.model.float()
        self.banlance_loss = Cross_modal_class_balance_loss(cfg["num_bit"]).to(
            cfg["device"]
        )

        self.all_dataloader, self.all_label = self.init_dataset()
        self.train_loader, self.query_loader, self.retrieval_loader = (
            self.all_dataloader
        )
        self.train_labels, self.query_labels, self.retrieval_labels = self.all_label

        self.train_labels, self.query_labels, self.retrieval_labels = (
            self.train_labels.to(cfg["device"]),
            self.query_labels.to(cfg["device"]),
            self.retrieval_labels.to(cfg["device"]),
        )

        self.num_train, self.num_query, self.num_retrieval = (
            self.train_labels.shape[0],
            self.query_labels.shape[0],
            self.retrieval_labels.shape[0],
        )

        self.train_log = f"dataset:{cfg['dataset']} class:{self.train_labels.shape[1]} dropout:{cfg['dropout']} train:{self.num_train} query:{self.num_query} retrieval:{self.num_retrieval} total:{self.num_query+self.num_retrieval}"
        self.rot_dict = {}
        for one in cfg["list_bit"]:
            self.rot_dict[one] = Rot(dim=one).to(cfg["device"])
        self.optimizer = DPDH_Adam(
            [
                {"params": self.model.clip.parameters(), "lr": cfg["clip_lr"]},
                {"params": self.model.image_pre.parameters(), "lr": cfg["other_lr"]},
                {"params": self.model.text_pre.parameters(), "lr": cfg["other_lr"]},
                {"params": self.model.FuseTrans.parameters(), "lr": cfg["other_lr"]},
            ],
            lr=0.001,
            warmup=0.1,
            schedule="warmup_cosine",
            b1=0.9,
            b2=0.98,
            e=1e-6,
            t_total=len(self.train_loader) * cfg["train_epoch"],
            weight_decay=0.2,
            max_grad_norm=1.0,
        )

        self.optimizer_loss = torch.optim.SGD(
            params=self.dpdh.parameters(), lr=0.00001, weight_decay=0.0005
        )
        self.optimizer_dsph = torch.optim.SGD(
            params=self.dsph.parameters(), lr=0.00001, weight_decay=0.0005
        )

    def train(self):
        print(self.train_log)
        logging.info(f"\n\n{self.train_log}")
        self.model.train()
        all_loss = max_i2t = max_t2i = 0
        distance = distances.CosineSimilarity()
        reducer = reducers.ThresholdReducer(low=0)
        config = get_config()
        mining_func = TripletAutoParamsMiner(
            distance=distance,
            margin_init=0.25,
            beta_init=0,
            type_of_triplets="semihard",
            k=2,
            k_n=2,
            k_p=2,
            mode="normal",
        )
        loss_matching_func = TripletCustomMarginLoss(
            margin=0.25, distance=distance, reducer=reducer
        )
        loss_id_func = LowerBoundLoss()
        F_buffer = G_buffer = L_buffer = Bx = By = {}
        for one in cfg["list_bit"]:
            F_buffer[one] = torch.randn(cfg["num_train"], one).to(
                cfg["device"], non_blocking=True
            )
            G_buffer[one] = torch.randn(cfg["num_train"], one).to(
                cfg["device"], non_blocking=True
            )
            L_buffer[one] = torch.randn(cfg["num_train"], one).to(
                cfg["device"], non_blocking=True
            )
            By[one] = torch.sign(G_buffer[one] + F_buffer[one])
        ALL_LOSS = {}
        for i in range(cfg["train_epoch"]):
            for step, (image, text, key_padding_mask, label, index) in enumerate(
                tqdm(self.train_loader)
            ):
                image, text, key_padding_mask, label = (
                    image.to(cfg["device"], non_blocking=True),
                    text.to(cfg["device"], non_blocking=True),
                    key_padding_mask.to(cfg["device"], non_blocking=True),
                    label.to(cfg["device"], non_blocking=True).float(),
                )
                index = index.numpy()
                out_dict = self.model(image, text, key_padding_mask, label)
                for one in cfg["list_bit"]:
                    loss = 0
                    hash_img = out_dict[f"image_hash_{one}"]
                    hash_text = out_dict[f"text_hash_{one}"]
                    hash_label = out_dict["label_features"]
                    img_pre = out_dict["image_pre"]
                    txt_pre = out_dict["text_pre"]

                    img_rot = F.normalize(self.rot_dict[one](hash_img.T).T)
                    text_rot = F.normalize(self.rot_dict[one](hash_text.T).T)
                    label_rot = F.normalize(self.rot_dict[64](hash_label.T).T)

                    criterion = bit_var_loss()
                    method = MetricLearningMethods(
                        config,
                        mining_func,
                        loss_matching=loss_matching_func,
                        loss_identity=loss_id_func,
                    )
                    q_img_loss = criterion(img_rot)
                    q_text_loss = criterion(text_rot)
                    q_label_loss = criterion(label_rot)
                    t_img_loss = method.calculate_total_loss(
                        img_rot, label, cfg["device"], epoch_id=i, batch_id=step
                    )
                    t_text_loss = method.calculate_total_loss(
                        text_rot, label, cfg["device"], epoch_id=i, batch_id=step
                    )
                    t_label_loss = method.calculate_total_loss(
                        label_rot, label, cfg["device"], epoch_id=i, batch_id=step
                    )

                    if cfg["dataset"] == "nuswide":
                        loss_xx = loss_xl = loss_yy = loss_y1 = 0
                    else:
                        loss_xx = multilabelsimilarity_loss(
                            label, self.train_labels, img_rot, G_buffer[one]
                        )
                        loss_xl = multilabelsimilarity_loss(
                            label, self.train_labels, img_rot, L_buffer[one]
                        )
                        loss_yy = multilabelsimilarity_loss(
                            label, self.train_labels, text_rot, G_buffer[one]
                        )
                        loss_y1 = multilabelsimilarity_loss(
                            label, self.train_labels, text_rot, L_buffer[one]
                        )
                    quantization_x1 = quantization_Loss(text_rot, By[one][index, :])
                    quantization_y1 = quantization_Loss(img_rot, By[one][index, :])

                    loss_dpdh = self.dpdh(img_rot, text_rot, label, img_pre, txt_pre)
                    loss_dsph = self.dsph(img_rot, text_rot, label)

                    loss_1 = (
                        t_text_loss
                        + q_text_loss
                        + q_label_loss
                        + q_img_loss
                        + t_img_loss
                        + t_label_loss
                    )
                    loss_2 = (
                        quantization_x1
                        + quantization_y1
                        + loss_xx
                        + loss_xl
                        + loss_yy
                        + loss_y1
                    )
                    loss = loss_dsph + loss_dpdh + loss_1 + loss_2
                    ALL_LOSS[one] = loss
                all_loss = 0
                for key in ALL_LOSS:
                    all_loss += ALL_LOSS[key]
                all_loss /= len(cfg["list_bit"])
                self.optimizer.zero_grad(set_to_none=True)
                self.optimizer_loss.zero_grad(set_to_none=True)
                self.optimizer_dsph.zero_grad(set_to_none=True)
                all_loss.backward()
                self.optimizer.step()
                self.optimizer_loss.step()
                self.optimizer_dsph.step()
            self.valid(i + 1, ALL_LOSS)

    def valid(self, e, all_loss):
        self.model.eval()
        query_i_buffer, query_t_buffer = self.encode(self.query_loader, self.num_query)
        retrieval_i_buffer, retrieval_t_buffer = self.encode(
            self.retrieval_loader, self.num_retrieval
        )
        for one in cfg["list_bit"]:
            mAPi2i = calc_map_k_matrix(
                query_i_buffer[one],
                retrieval_i_buffer[one],
                self.query_labels,
                self.retrieval_labels,
            )

            mAPi2t = calc_map_k_matrix(
                query_i_buffer[one],
                retrieval_t_buffer[one],
                self.query_labels,
                self.retrieval_labels,
            )
            mAPt2i = calc_map_k_matrix(
                query_t_buffer[one],
                retrieval_i_buffer[one],
                self.query_labels,
                self.retrieval_labels,
            )
            radius = calc_precisions_hamming_radius(
                query_i_buffer[one],
                retrieval_t_buffer[one],
                self.query_labels,
                self.retrieval_labels,
            )

            # p_i2t, r_i2t = pr_curve(
            #     query_i_buffer[one],
            #     retrieval_t_buffer[one],
            #     self.query_labels,
            #     self.retrieval_labels,
            # )
            # p_i2t, r_i2t = pr_curve(
            #     query_i_buffer[one],
            #     retrieval_t_buffer[one],
            #     self.query_labels,
            #     self.retrieval_labels,
            # )
            # p_t2i, r_t2i = pr_curve(
            #     query_t_buffer[one],
            #     retrieval_i_buffer[one],
            #     self.query_labels,
            #     self.retrieval_labels,
            # )
            # topN_i2t = p_top(
            #     query_i_buffer[one],
            #     retrieval_t_buffer[one],
            #     self.query_labels,
            #     self.retrieval_labels,
            # )
            # topN_t2i = p_top(
            #     query_t_buffer[one],
            #     retrieval_i_buffer[one],
            #     self.query_labels,
            #     self.retrieval_labels,
            # )
            # ng_i2t_1000 = compute_ndcg_at_n(
            #     query_i_buffer[one],
            #     retrieval_t_buffer[one],
            #     self.query_labels,
            #     self.retrieval_labels,
            # )
            # ng_t2i_1000 = compute_ndcg_at_n(
            #     query_t_buffer[one],
            #     retrieval_i_buffer[one],
            #     self.query_labels,
            #     self.retrieval_labels,
            # )

            print(
                f"{str(e).zfill(2)}/{cfg['train_epoch']} {one}bit all_loss:{all_loss[one]:.4f} mAPi2t:{mAPi2t:.4f} mAPt2i:{mAPt2i:.4f} mAPi2i:{mAPi2i:.4f} radius:{radius}"
            )
            # print(p_i2t, r_i2t)
            # print(p_t2i, r_t2i)
            # print(topN_i2t)
            # print(topN_t2i)
            # print(ng_i2t_1000)
            # print(ng_t2i_1000)
            logging.info(
                f"{str(e).zfill(2)}/{cfg['train_epoch']} {one}bit all_loss:{all_loss[one]:.4f} mAPi2t:{mAPi2t:.4f} mAPt2i:{mAPt2i:.4f} mAPi2i:{mAPi2i:.4f} radius:{radius}"
            )
        # query_img, query_txt = self.encode(self.query_loader, self.num_query)
        # retrieval_img, retrieval_txt = self.encode(
        #     self.retrieval_loader, self.num_retrieval
        # )
        # mAPi2t = calc_map_k_matrix(
        #     query_img, retrieval_txt, self.query_labels, self.retrieval_labels
        # )
        # mAPt2i = calc_map_k_matrix(
        #     query_txt, retrieval_img, self.query_labels, self.retrieval_labels
        # )

        # train_mAPi2t = calc_map_k_matrix(query_img, train_retrieval_img, query_labels, train_labels)
        # train_mAPt2i = calc_map_k_matrix(query_txt, train_retrieval_txt, query_labels, train_labels)
        # p, r = pr_curve(query_img, retrieval_txt, query_labels, retrieval_labels)
        # top_value = p_top(query_img, retrieval_txt, query_labels, retrieval_labels)
        # curve_pr(p,r)
        # print(top_value)
        # ng_1000 = compute_ndcg_at_n(query_img, retrieval_txt, query_labels, retrieval_labels)
        # print(top_value)
        return mAPi2t, mAPt2i, 1, 1

    def encode(self, data_loader, length):
        i_buffer = {}
        t_buffer = {}
        for i, one in enumerate(cfg["list_bit"]):
            i_buffer[one] = torch.empty(length, one, dtype=torch.float).to(
                cfg["device"]
            )
            t_buffer[one] = torch.empty(length, one, dtype=torch.float).to(
                cfg["device"]
            )
        for image, text, padding_mask, label, index in tqdm(data_loader):
            image, text, label = (
                image.to(cfg["device"], non_blocking=True),
                text.to(cfg["device"], non_blocking=True),
                label.to(cfg["device"], non_blocking=True).float(),
            )
            index = index.numpy()
            out_dict = self.model(image, text, padding_mask, label)
            for one in cfg["list_bit"]:
                i_buffer[one][index, :] = torch.sign(out_dict[f"image_hash_{one}"].data)
                t_buffer[one][index, :] = torch.sign(out_dict[f"text_hash_{one}"].data)

        return i_buffer, t_buffer

        # img_buffer = torch.empty(length, cfg["num_bit"], dtype=torch.float).to(
        #     cfg["device"]
        # )
        # text_buffer = torch.empty(length, cfg["num_bit"], dtype=torch.float).to(
        #     cfg["device"]
        # )
        # for image, text, padding_mask, label, index in tqdm(data_loader):
        #     image, text, label = (
        #         image.to(cfg["device"], non_blocking=True),
        #         text.to(cfg["device"], non_blocking=True),
        #         label.to(cfg["device"], non_blocking=True).float(),
        #     )
        #     index = index.numpy()
        #     # hash_img, hash_text, _, _, _ = self.model(image, text, padding_mask, label)
        #     out_dict = self.model(image, text, padding_mask, label)
        #     hash_img = out_dict["image_hash_64"]
        #     hash_text = out_dict["text_hash_64"]
        #     hash_img = torch.sign(hash_img.detach())
        #     hash_text = torch.sign(hash_text.detach())
        #     img_buffer[index, :] = hash_img.data
        #     text_buffer[index, :] = hash_text.data
        # return img_buffer, text_buffer

    def init_dataset(self):
        index_file = os.path.join(
            cfg["project_root"], "data_mat", cfg["dataset"], cfg["index_file"]
        )
        caption_file = os.path.join(
            cfg["project_root"], "data_mat", cfg["dataset"], cfg["caption_file"]
        )
        label_file = os.path.join(
            cfg["project_root"], "data_mat", cfg["dataset"], cfg["label_file"]
        )
        train_data, query_data, retrieval_data = generate_dataset(
            captionFile=caption_file,
            indexFile=index_file,
            labelFile=label_file,
            dataset_name=cfg["dataset"],
            maxWords=cfg["max_words"],
            imageResolution=cfg["image_resolution"],
            query_num=cfg["num_query"],
            train_num=cfg["num_train"],
            seed=cfg["seed"],
        )

        train_labels = train_data.get_all_label().float()
        query_labels = query_data.get_all_label().float()
        retrieval_labels = retrieval_data.get_all_label().float()

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=cfg["train_batch_size"],
            num_workers=cfg["num_workers"],
            pin_memory=True,
            shuffle=True,
            prefetch_factor=2,
        )
        query_loader = DataLoader(
            dataset=query_data,
            batch_size=cfg["query_batch_size"],
            num_workers=cfg["num_workers"],
            pin_memory=True,
            shuffle=True,
            prefetch_factor=2,
        )
        retrieval_loader = DataLoader(
            dataset=retrieval_data,
            batch_size=cfg["retrieval_batch_size"],
            num_workers=cfg["num_workers"],
            pin_memory=True,
            shuffle=True,
            prefetch_factor=2,
        )

        return (train_loader, query_loader, retrieval_loader), (
            train_labels,
            query_labels,
            retrieval_labels,
        )


if __name__ == "__main__":
    WYL_Trainer().train()
