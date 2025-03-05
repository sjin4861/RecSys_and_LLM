# model_loader.py
import argparse
import json

import torch
from huggingface_hub import hf_hub_download

from recsys_and_llm.ml.models.ALLMRec.a_llmrec_model import A_llmrec_model
from recsys_and_llm.ml.models.gSASRec.gsasrec_inference import build_model
from recsys_and_llm.ml.models.TiSASRec.TiSASRec_inference import TiSASRec


class ModelLoader:
    def __init__(self, llmrec_args):
        self.llmrec_args = llmrec_args
        self.allmrec_model = None
        self.tisasrec_model = None
        self.gsasrec_model = None
        self.tisasrec_args = None
        self.gsasrec_args = None

        self._load_models()

    def _load_allmrec(self):
        """ALLMRec 모델 로드 및 초기화"""
        self.allmrec_model = A_llmrec_model(self.llmrec_args).to(
            self.llmrec_args.device
        )
        phase1_epoch = 10
        phase2_epoch = 10
        self.allmrec_model.load_model(
            self.llmrec_args, phase1_epoch=phase1_epoch, phase2_epoch=phase2_epoch
        )
        self.allmrec_model.eval()

    def _load_tisasrec(self):
        """TiSASRec 모델 로드"""
        repo_id = "PNUDI/TiSASRec"
        model_file = hf_hub_download(
            repo_id=repo_id, filename="pytorch_model.bin", repo_type="model"
        )
        config_file = hf_hub_download(
            repo_id=repo_id, filename="config.json", repo_type="model"
        )
        with open(config_file, "r") as f:
            config_data = json.load(f)
        self.tisasrec_args = argparse.Namespace(**config_data)
        self.tisasrec_model = TiSASRec(
            self.tisasrec_args.usernum,
            self.tisasrec_args.itemnum,
            self.tisasrec_args.itemnum,
            self.tisasrec_args,
        ).to(self.tisasrec_args.device)
        self.tisasrec_model.load_state_dict(
            torch.load(model_file, map_location=self.tisasrec_args.device)
        )
        self.tisasrec_model.eval()

    def _load_gsasrec(self):
        """gSASRec 모델 로드"""
        repo_id = "PNUDI/gSASRec"
        model_file = hf_hub_download(
            repo_id=repo_id, filename="pytorch_model.bin", repo_type="model"
        )
        config_file = hf_hub_download(
            repo_id=repo_id, filename="config.json", repo_type="model"
        )
        with open(config_file, "r") as f:
            config_data = json.load(f)
        self.gsasrec_args = argparse.Namespace(**config_data)
        self.gsasrec_model = build_model(self.gsasrec_args)
        self.gsasrec_model.load_state_dict(torch.load(model_file, map_location="cpu"))
        self.gsasrec_model.eval()

    def _load_contentrec(self):
        """모델 로드"""
        self.item_contents_emb = torch.load(
            hf_hub_download(
                repo_id="PNUDI/Item_based",
                filename="item_text_emb_SB.pt",
                repo_type="model",
            ),
            weights_only=False,
        )

    def _load_models(self):
        self._load_allmrec()
        self._load_tisasrec()
        self._load_gsasrec()
        self._load_contentrec()

    def get_models(self):
        return {
            "allmrec_model": self.allmrec_model,
            "tisasrec_model": self.tisasrec_model,
            "gsasrec_model": self.gsasrec_model,
            "contentrec_model": self.item_contents_emb,
            "tisasrec_args": self.tisasrec_args,
            "gsasrec_args": self.gsasrec_args,
        }
