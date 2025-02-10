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
custom_prompt = {
    "role": "system",
    "content": """
    당신은 사용자에게 영화를 추천하는 시스템입니다. 
    반드시 영어로 답변해야 하며, 
    **5개의** 추천 영화를 제시해야 합니다.
    답변의 출력 형식은 
    **영화제목(개봉연도)으로 제한합니다. 그 외의 내용(기호, 특수문자, 기타 용어 등)은 작성하지 마세요.**
    **하지만, 사용자가 추천 이유나 추천 영화에 관한 정보를 물어 볼 때에는 예외로 합니다.**
    사용자의 선호에 관한 정보(배우, 감독, 장르) 등이 부족할 경우, 필요한 만큼 대화를 이어 나갈 수 있습니다.
    사용자가 대화 종료 신호를 보일 경우, 적절한 인삿말로 대화를 마무리하세요.
    """
}

##################
# Recommendation #
##################
unicrs_rec = UnicrsRec.from_pretrained("recwizard/unicrs-rec-redial")
redial_rec = RedialRec.from_pretrained('recwizard/redial-rec')
gpt_rec_fillblank = ChatgptRec.from_pretrained('recwizard/chatgpt-rec-fillblank')
##################
#   Generation   #
##################
unicrs_gen = UnicrsGen.from_pretrained("recwizard/unicrs-gen-redial")
gpt_gen_expansion = ChatgptGen.from_pretrained(
    model_name='gpt-4o',
    pretrained_model_name_or_path='recwizard/chatgpt-gen-expansion',
    prompt=custom_prompt
)
gpt_gen_fillblank = ChatgptGen.from_pretrained(
    model_name='gpt-4o',
    pretrained_model_name_or_path='recwizard/chatgpt-gen-fillblank',
    prompt=custom_prompt
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
            config=FillBlankConfig(), rec_module=redial_rec, gen_module=gpt_gen_fillblank
        )
    elif flag.lower() == "expansion":
        return ExpansionPipeline(
            config=ExpansionConfig(), rec_module=unicrs_rec, gen_module=gpt_gen_expansion
        )
    elif flag.lower() == "gpt":
        return FillBlankPipeline(
            config=FillBlankConfig(), rec_module=gpt_rec_fillblank, gen_module=gpt_gen_fillblank
        )
    elif flag.lower() == "default":
        return FillBlankPipeline(
            config=FillBlankConfig(), rec_module=unicrs_rec, gen_module=unicrs_gen
        )
    return "DefaultUniCRS"
