# app/components/pipeline_manager.py
import os

from dotenv import load_dotenv
from recwizard import (
    ChatgptGen,
    ChatgptRec,
    ExpansionConfig,
    ExpansionPipeline,
    FillBlankConfig,
    FillBlankPipeline,
    RedialRec,
    UnicrsGen,
    UnicrsRec,
    ChatgptRec,
)

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
custom_prompt = {
    "role": "system",
    "content": """
    You are a system that recommends movies to users.
    You must answer in English,
    In the answer, the output format of Entity is
    **Limited to the movie title (opening year). **
    **However, exceptions are made when users ask for recommendation reasons or information about recommended movies.**
    If you lack information about your preferences (actors, directors, genres), etc., you can continue the conversation as much as you need.
    If the user shows signs of starting or ending the conversation, end the conversation with an appropriate greeting.
    """
}

##################
# Recommendation #
##################
unicrs_rec = UnicrsRec.from_pretrained("recwizard/unicrs-rec-redial")
redial_rec = RedialRec.from_pretrained("recwizard/redial-rec")
gpt_rec_fillblank = ChatgptRec.from_pretrained("recwizard/chatgpt-rec-fillblank")
##################
#   Generation   #
##################
unicrs_gen = UnicrsGen.from_pretrained("recwizard/unicrs-gen-redial")
gpt_gen_expansion = ChatgptGen.from_pretrained(
    model_name="gpt-4o",
    pretrained_model_name_or_path="recwizard/chatgpt-gen-expansion",
    prompt=custom_prompt,
)
gpt_gen_fillblank = ChatgptGen.from_pretrained(
    model_name="gpt-4o",
    pretrained_model_name_or_path="recwizard/chatgpt-gen-fillblank",
    prompt=custom_prompt,
)
##################
# Configurations #
##################
fill_blank_config = FillBlankConfig()
expansion_config = ExpansionConfig()


def set_pipeline(flag, rec_module=unicrs_rec, gen_module=gpt_gen_fillblank):
    """
    flag, rec_module, gen_module에 따라 적절한 Pipeline 객체 생성.
    """
    if flag.lower() == "blank":
        return FillBlankPipeline(
            config=FillBlankConfig(),
            rec_module=redial_rec,
            gen_module=gpt_gen_fillblank,
        )
    elif flag.lower() == "expansion":
        return ExpansionPipeline(
            config=ExpansionConfig(),
            rec_module=unicrs_rec,
            gen_module=gpt_gen_expansion,
        )
    elif flag.lower() == "gpt":
        return FillBlankPipeline(
            config=FillBlankConfig(),
            rec_module=gpt_rec_fillblank,
            gen_module=gpt_gen_fillblank,
        )
    elif flag.lower() == "default":
        return FillBlankPipeline(
            config=FillBlankConfig(), rec_module=unicrs_rec, gen_module=unicrs_gen
        )
    return "DefaultUniCRS"
