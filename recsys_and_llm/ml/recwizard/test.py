import os

import huggingface_hub
import recwizard

os.environ["OPENAI_API_KEY"] = (
    "OPENAI API 키 입력해주시면 됩니다 : gpt 기반 생성 모델의 경우 현재 이슈로 인하여 계획 중지되었습니다!"
)
# os.environ["HUGGINGFACE_TOKEN"] = "허깅페이스 계정 토큰 입력해주시면 됩니다"

from recwizard import (
    ChatgptGen,
    ExpansionConfig,
    ExpansionPipeline,
    FillBlankConfig,
    FillBlankPipeline,
    LlamaGen,
    RedialRec,
    UnicrsGen,
    UnicrsRec,
)

# huggingface_hub.login(token=huggingface_token, add_to_git_credential=False)
# 별도의 프롬프트 예시 문장을 선언해 두었습니다.
# 다만, 프롬프트 입력을 어떠한 부분에 해야 하는지에 대해서는 소스코드를 통한 분석 진행 중 단계입니다ㅜㅜ
# prompt = "You are a system that provides movie recommendations to users. Users will be asked to recommend their top 5 favorite movies, and can continue the conversation further if needed."

##################
# Recommendation #
#     Modules    #
##################
unicrs_rec = UnicrsRec.from_pretrained("recwizard/unicrs-rec-redial")
# kgsf_rec = KgsfRec.from_pretrained('recwizard/kgsf-rec')

##################
#   Generation   #
#    Modules     #
##################
# unicrs_gen = UnicrsGen.from_pretrained('recwizard/unicrs-gen-redial')
# kgsf_gen = KgsfGen.from_pretrained('recwizard/kgsf-gen')
"""
        Initializes the instance based on the config file.

        Args:
            config (ChatgptGenConfig): The config file.
            prompt (str, optional): A prompt to override the prompt from config file.
            model_name (str, optional): The specified GPT model's name. 
"""
gpt_gen = ChatgptGen.from_pretrained(
    pretrained_model_name_or_path="recwizard/chatgpt-gen-expansion"
)
# llama_gen = LlamaGen.from_pretrained('recwizard/llama-expansion')
##################
#  Configuration #
#     Setups     #
##################

"""
class FillBlankConfig(BaseConfig):
    def __init__(self, rec_pattern: str = r"<movie>", resp_prompt="System:", **kwargs):
        super().__init__(**kwargs)
        self.rec_pattern = rec_pattern
        self.resp_prompt = resp_prompt
"""
# gpt_config = ChatgptAgentConfig()
fill_blank_config = FillBlankConfig(resp_prompt=prompt)
expansion_config = ExpansionConfig(prompt=prompt)


"""
    set_pipeline : 모듈을 활용한 pipeline setup 함수
    
    Params.
    flag : 'Blank'로 설정 시 FillBlankPipeline, 아닐 경우 ExpansionPipeline
    rec_module : 추천 모듈 변수명 입력
    gen_module : 응답 생성 모듈 변수명 입력
"""


def set_pipeline(flag, rec_module, gen_module):
    if flag.lower() == "blank":
        pipeline = FillBlankPipeline(
            config=fill_blank_config, rec_module=rec_module, gen_module=gen_module
        )
    else:
        pipeline = recwizard.ExpansionPipeline(
            config=expansion_config, rec_module=rec_module, gen_module=gen_module
        )
    return pipeline


def chat_with_system(pipeline):
    context = []
    print(context)
    print("대화를 시작합니다. 종료하려면 'quit'를 입력하세요.")

    while True:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break

        # 사용자 입력을 컨텍스트에 추가
        context.append(f"User: {user_input}")

        # 전체 컨텍스트를 <sep>로 연결
        full_context = "<sep>".join(context)

        # 응답 생성
        response = pipeline.response(full_context)
        print(response)

        # 시스템 응답을 컨텍스트에 추가
        context.append(f"{response}")
        full_context = "<sep>".join(context)
        print(full_context)


def test_single_turn(pipeline):
    # 대화 컨텍스트 설정
    context = "<sep>".join(
        [
            "User: Hello!",
            "System: Hello, I have some movie ideas for you. Have you watched the movie <entity>Interstella</entity>?",
            "User: Yes, i've seen that movie. And i'm Looking for other movies in the Romance category.",
        ]
    )
    print(pipeline.response(context))
    print(context)


if __name__ == "__main__":
    pipeline = set_pipeline("Expansion", unicrs_rec, gpt_gen)
    # test_single_turn(pipeline)
    chat_with_system(pipeline)
