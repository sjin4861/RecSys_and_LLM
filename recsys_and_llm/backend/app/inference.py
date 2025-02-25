# inference.py
import numpy as np

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
