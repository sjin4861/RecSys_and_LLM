# app/components/pipeline_manager.py
import os

from dotenv import load_dotenv
from recwizard import (
    ChatgptGen,
    ExpansionConfig,
    ExpansionPipeline,
    FillBlankConfig,
    FillBlankPipeline,
    RedialRec,
    UnicrsGen,
    UnicrsRec,
)

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
##################
# Recommendation #
##################
unicrs_rec = UnicrsRec.from_pretrained("recwizard/unicrs-rec-redial")
##################
#   Generation   #
##################
unicrs_gen = UnicrsGen.from_pretrained("recwizard/unicrs-gen-redial")
gpt_gen = ChatgptGen.from_pretrained("RecWizard/chatgpt-gen-fillblank")
##################
# Configurations #
##################
fill_blank_config = FillBlankConfig()
expansion_config = ExpansionConfig()


def set_pipeline(flag, rec_module=unicrs_rec, gen_module=gpt_gen):
    """
    flag, rec_module, gen_module에 따라 적절한 Pipeline 객체 생성.
    """
    if flag.lower() == "blank":
        return FillBlankPipeline(
            config=FillBlankConfig(), rec_module=rec_module, gen_module=gen_module
        )
    elif flag.lower() == "expansion":
        return ExpansionPipeline(
            config=ExpansionConfig(), rec_module=rec_module, gen_module=gen_module
        )
    elif flag.lower() == "gpt":
        return FillBlankPipeline(
            config=FillBlankConfig(), rec_module=rec_module, gen_module=gen_module
        )
    elif flag.lower() == "default":
        return FillBlankPipeline(
            config=FillBlankConfig(), rec_module=unicrs_rec, gen_module=unicrs_gen
        )
    return "DefaultUniCRS"
