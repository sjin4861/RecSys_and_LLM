import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from models.llm4rec import *
from models.recsys_model import *
from pre_train.sasrec.utils import *
from sentence_transformers import SentenceTransformer
from torch.cuda.amp import autocast as autocast


class two_layer_mlp(nn.Module):
    def __init__(self, dims):
        super().__init__()
        # fc1: encoder, fc2: decoder
        self.fc1 = nn.Linear(dims, 128)
        self.fc2 = nn.Linear(128, dims)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x1 = self.fc2(x)
        return x, x1


class A_llmrec_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        rec_pre_trained_data = args.rec_pre_trained_data
        self.args = args
        self.device = args.device

        with open(
            f"./data/amazon/{args.rec_pre_trained_data}_text_name_dict.json.gz", "rb"
        ) as ft:
            self.text_name_dict = pickle.load(ft)

        # pre_trained 모델 불러오기
        self.recsys = RecSys(args.recsys, rec_pre_trained_data, self.device)
        self.item_num = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units
        self.sbert_dim = 768

        # Item Encoder
        self.mlp = two_layer_mlp(self.rec_sys_dim)
        if args.pretrain_stage1:
            self.sbert = SentenceTransformer("nq-distilbert-base-v1")
            # Text Encoder
            self.mlp2 = two_layer_mlp(self.sbert_dim)

        # - Encoder -> Item & Text Embedding을 align 해준다 -> Matching Loss 계산
        # - 기존 Embedding & Encoder-Decoder 거친 Embedding -> Reconstructuion Loss 계산
        self.mse = nn.MSELoss()

        self.maxlen = args.maxlen
        self.NDCG = 0
        self.HIT = 0
        self.rec_NDCG = 0
        self.rec_HIT = 0
        self.lan_NDCG = 0
        self.lan_HIT = 0
        self.num_user = 0
        self.yes = 0

        # User Representation & 긍정 샘플 / 부정 샘플 -> Recommendation Loss(BPR Loss) 계산
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()

        if args.pretrain_stage2 or args.inference:
            self.llm = llm4rec(device=self.device, llm_model=args.llm)

            self.log_emb_proj = nn.Sequential(
                nn.Linear(self.rec_sys_dim, self.llm.llm_model.config.hidden_size),
                nn.LayerNorm(self.llm.llm_model.config.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(
                    self.llm.llm_model.config.hidden_size,
                    self.llm.llm_model.config.hidden_size,
                ),
            )
            nn.init.xavier_normal_(self.log_emb_proj[0].weight)
            nn.init.xavier_normal_(self.log_emb_proj[3].weight)

            self.item_emb_proj = nn.Sequential(
                nn.Linear(128, self.llm.llm_model.config.hidden_size),
                nn.LayerNorm(self.llm.llm_model.config.hidden_size),
                nn.GELU(),
                nn.Linear(
                    self.llm.llm_model.config.hidden_size,
                    self.llm.llm_model.config.hidden_size,
                ),
            )
            nn.init.xavier_normal_(self.item_emb_proj[0].weight)
            nn.init.xavier_normal_(self.item_emb_proj[3].weight)

    def save_model(self, args, epoch1=None, epoch2=None):
        out_dir = f"./models/saved_models/"
        create_dir(out_dir)
        out_dir += f"{args.rec_pre_trained_data}_{args.recsys}_{epoch1}_"
        if args.pretrain_stage1:
            torch.save(self.sbert.state_dict(), out_dir + "sbert.pt")
            torch.save(self.mlp.state_dict(), out_dir + "mlp.pt")
            torch.save(self.mlp2.state_dict(), out_dir + "mlp2.pt")

        out_dir += f"{args.llm}_{epoch2}_"
        if args.pretrain_stage2:
            torch.save(self.log_emb_proj.state_dict(), out_dir + "log_proj.pt")
            torch.save(self.item_emb_proj.state_dict(), out_dir + "item_proj.pt")

    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        out_dir = f"./models/saved_models/{args.rec_pre_trained_data}_{args.recsys}_{phase1_epoch}_"

        mlp = torch.load(out_dir + "mlp.pt", map_location=args.device)
        self.mlp.load_state_dict(mlp)
        del mlp
        for name, param in self.mlp.named_parameters():
            param.requires_grad = False

        if args.inference:
            out_dir += f"{args.llm}_{phase2_epoch}_"

            log_emb_proj_dict = torch.load(
                out_dir + "log_proj.pt", map_location=args.device
            )
            self.log_emb_proj.load_state_dict(log_emb_proj_dict)
            del log_emb_proj_dict

            item_emb_proj_dict = torch.load(
                out_dir + "item_proj.pt", map_location=args.device
            )
            self.item_emb_proj.load_state_dict(item_emb_proj_dict)
            del item_emb_proj_dict

    def find_item_text(self, item, title_flag=True, description_flag=True):
        t = "title"
        d = "description"
        t_ = "No Title"
        d_ = "No Description"
        if title_flag and description_flag:
            return [
                f'"{self.text_name_dict[t].get(i,t_)}, {self.text_name_dict[d].get(i,d_)}"'
                for i in item
            ]
        elif title_flag and not description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}"' for i in item]
        elif not title_flag and description_flag:
            return [f'"{self.text_name_dict[d].get(i,d_)}"' for i in item]

    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        t = "title"
        d = "description"
        t_ = "No Title"
        d_ = "No Description"
        if title_flag and description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}, {self.text_name_dict[d].get(item,d_)}"'
        elif title_flag and not description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}"'
        elif not title_flag and description_flag:
            return f'"{self.text_name_dict[d].get(item,d_)}"'

    def get_item_emb(self, item_ids):
        with torch.no_grad():
            item_embs = self.recsys.model.item_emb(
                torch.LongTensor(item_ids).to(self.device)
            )
            item_embs, _ = self.mlp(item_embs)

        return item_embs

    def forward(self, data, optimizer=None, batch_iter=None, mode="phase1"):
        if mode == "phase1":
            self.pre_train_phase1(data, optimizer, batch_iter)
        if mode == "phase2":
            self.pre_train_phase2(data, optimizer, batch_iter)
        if mode == "generate":
            self.generate(data)

    def make_interact_text(self, interact_ids, interact_max_num):
        interact_item_titles_ = self.find_item_text(
            interact_ids, title_flag=True, description_flag=False
        )
        interact_text = []
        if interact_max_num == "all":
            for title in interact_item_titles_:
                interact_text.append(title + "[HistoryEmb]")
        else:
            for title in interact_item_titles_[-interact_max_num:]:
                interact_text.append(title + "[HistoryEmb]")
            interact_ids = interact_ids[-interact_max_num:]

        interact_text = ",".join(interact_text)
        return interact_text, interact_ids

    def make_candidate_text(self, interact_ids, candidate_num):
        candidate_ids = make_candidate_for_LLM(
            self.recsys.model, self.item_num, interact_ids, self.args
        )
        candidate_text = []

        # candidate_ids에 neg_item_id(랜덤 생성)를 넣는다,,,,, -> historical & user-representation을 결합한 벡터와 유사한 아이템을 넣을 순 없나
        for candidate in candidate_ids:
            candidate_text.append(
                self.find_item_text_single(
                    candidate, title_flag=True, description_flag=False
                )
                + "[CandidateEmb]"
            )

        random_ = np.random.permutation(len(candidate_text))
        candidate_text = np.array(candidate_text)[random_]
        candidate_ids = np.array(candidate_ids)[random_]

        return ",".join(candidate_text), candidate_ids

    def generate(self, data):
        u, seq, neg, rank = data

        answer = []
        text_input = []
        interact_embs = []
        candidate_embs = []
        with torch.no_grad():
            log_emb = self.recsys.model(u, seq, neg)

            interact_text, interact_ids = self.make_interact_text(seq[seq > 0], 10)

            candidate_num = 20
            candidate_text, candidate_ids = self.make_candidate_text(
                seq[seq > 0], candidate_num
            )

            input_text = ""
            input_text += " is a user representation."
            if self.args.rec_pre_trained_data == "Movies_and_TV":
                input_text += "This user has watched "
            elif self.args.rec_pre_trained_data == "Video_Games":
                input_text += "This user has played "
            elif (
                self.args.rec_pre_trained_data == "Luxury_Beauty"
                or self.args.rec_pre_trained_data == "Toys_and_Games"
            ):
                input_text += "This user has bought "

            input_text += interact_text

            if self.args.rec_pre_trained_data == "Movies_and_TV":
                input_text += " in the previous. Recommend one next movie for this user to watch next from the following movie title set, "
            elif self.args.rec_pre_trained_data == "Video_Games":
                input_text += " in the previous. Recommend one next game for this user to play next from the following game title set, "
            elif (
                self.args.rec_pre_trained_data == "Luxury_Beauty"
                or self.args.rec_pre_trained_data == "Toys_and_Games"
            ):
                input_text += " in the previous. Recommend one next item for this user to buy next from the following item title set, "

            input_text += candidate_text
            input_text += ". The recommendation is "

            text_input.append(input_text)

            interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))
            candidate_embs.append(self.item_emb_proj(self.get_item_emb(candidate_ids)))

        # user representation projection
        log_emb = self.log_emb_proj(log_emb)
        atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device)
        atts_llm = atts_llm.unsqueeze(1)
        log_emb = log_emb.unsqueeze(1)

        with torch.no_grad():
            self.llm.llm_tokenizer.padding_side = "left"
            llm_tokens = self.llm.llm_tokenizer(
                text_input, padding="longest", return_tensors="pt"
            ).to(self.device)

            with torch.cuda.amp.autocast():
                inputs_embeds = self.llm.llm_model.get_input_embeddings()(
                    llm_tokens.input_ids
                )

                llm_tokens, inputs_embeds = self.llm.replace_hist_candi_token(
                    llm_tokens, inputs_embeds, interact_embs, candidate_embs
                )

                attention_mask = llm_tokens.attention_mask
                # user representation + text input embedding
                inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

                outputs = self.llm.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=False,
                    top_p=0.9,
                    temperature=1,
                    num_beams=1,
                    max_length=512,
                    min_length=1,
                    pad_token_id=self.llm.llm_tokenizer.eos_token_id,
                    repetition_penalty=1.5,
                    length_penalty=1,
                    num_return_sequences=1,
                )

            outputs[outputs == 0] = 2  # convert output id 0 to 2 (eos_token_id)
            output_text = self.llm.llm_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]

        print(text_input[0])
        print("LLM: " + str(output_text[0]))

        return output_text
