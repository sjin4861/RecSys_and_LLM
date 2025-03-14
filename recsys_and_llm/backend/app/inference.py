# inference.py
import numpy as np
import torch
import torch.nn.functional as F

from recsys_and_llm.ml.models.gSASRec.gsasrec_inference import gsasrec_recommend_top5
from recsys_and_llm.ml.models.TiSASRec.TiSASRec_inference import tisasrec_recommend_top5
from recsys_and_llm.ml.utils import seq_preprocess


def inference(model_manager, user_id, seq, seq_time):
    # ALLMRec 기반 추천
    seq = seq_preprocess(model_manager.llmrec_args.maxlen, seq)
    seq = np.expand_dims(np.array(seq), axis=0)
    allmrec_prediction = model_manager.allmrec_model(seq, mode="inference")

    # TiSASRec 기반 추천
    tisasrec_prediction = tisasrec_recommend_top5(
        model_manager.tisasrec_args,
        model_manager.tisasrec_model,
        user_id,
        seq_time,
        model_manager.missing_list,
    )

    # gSASRec 기반 추천
    gsasrec_prediction = gsasrec_recommend_top5(
        model_manager.gsasrec_model,
        user_id,
        seq_time,
        model_manager.gsasrec_args,
        model_manager.missing_list,
    )

    return {
        "allmrec_prediction": allmrec_prediction,
        "tisasrec_prediction": tisasrec_prediction,
        "gsasrec_prediction": gsasrec_prediction,
    }


def item_content_inference(model_manager, item_id):
    item_contents_emb = torch.tensor(
        model_manager.contentrec_model, device=model_manager.device
    )
    target_emb = item_contents_emb[int(item_id) - 1]
    cosine_sim = F.cosine_similarity(target_emb, item_contents_emb)
    topk = torch.topk(cosine_sim, 21).indices  # 0~
    prediction = []

    for ele in topk:
        ele = ele.item() + 1
        if ele not in model_manager.missing_list and ele != int(item_id):
            prediction.append(str(ele))

    return prediction[:8]
