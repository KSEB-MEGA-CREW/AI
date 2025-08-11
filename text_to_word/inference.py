import warnings
# 'torch' 라이브러리를 임포트합니다. 파이토치는 딥러닝 모델을 구축하고 학습시키는 데 사용되는 핵심 프레임워크입니다.
import torch
# 'transformers' 라이브러리에서 필요한 클래스들을 임포트합니다.
from transformers import (
    # 'AutoTokenizer'는 사전 학습된 모델에 맞는 토크나이저를 자동으로 로드하는 클래스입니다.
    AutoTokenizer,
    # 'AutoModelForSeq2SeqLM'은 텍스트-투-텍스트(번역, 요약 등) 과업을 위한 사전 학습된 모델을 자동으로 로드하는 클래스입니다.
    AutoModelForSeq2SeqLM,
)
import config

warnings.filterwarnings("ignore", message="The following device_map keys do not match any submodules in the model:.*")

OUTPUT_DIR = config.OUTPUT_DIR

# 입력 시퀀스(한국어 문장)의 최대 길이를 128 토큰으로 설정합니다. 이보다 길면 잘라냅니다.
max_input_length = 128
# 타겟 시퀀스(수어 단어)의 최대 길이를 128 토큰으로 설정합니다.
max_target_length = 128

# 스크립트의 메인 로직을 포함하는 'main' 함수를 정의합니다.
def inference(input :str):
    # 10. 추론(Inference) 예시
    print("\n--- 추론 테스트 ---")

    # 'AutoTokenizer.from_pretrained'를 사용하여 저장된 토크나이저를 다시 불러옵니다.
    trained_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    
    # 'AutoModelForSeq2SeqLM.from_pretrained'를 사용하여 방금 저장한 파인튜닝된 모델을 다시 불러옵니다.
    trained_model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR, device_map="auto", ignore_mismatched_sizes=True)
    # trained_model.to(device)
    trained_model.eval()

    # 번역(gloss 생성)을 테스트할 한국어 문장을 정의합니다.
    text_to_translate = input
    # 'print' 함수와 f-string으로 어떤 문장이 입력되었는지 출력합니다.
    print(f"입력 문장: {text_to_translate}")

    # 입력 문장을 토큰화합니다. 'return_tensors="pt"'는 결과를 파이토치 텐서로 반환하라는 의미입니다. '.to(trained_model.device)'는 모델이 있는 장치(CPU 또는 GPU)로 텐서를 이동시킵니다.
    inputs = trained_tokenizer(text_to_translate, return_tensors="pt", max_length=max_input_length, truncation=True).to(trained_model.device)
    
    # if there are 'token_type_ids in the inputs, remove that
    if 'token_type_ids' in inputs:
        inputs.pop('token_type_ids')
        # print(f"input에서 token_type_ids가 제거되었습니다.")

    # 'trained_model.generate()' 메소드를 호출하여 입력 토큰(**inputs)으로부터 새로운 텍스트(gloss)를 생성합니다.
    # outputs = trained_model.generate(**inputs, max_length=max_target_length)
    with torch.no_grad():
        outputs = trained_model.generate(
            **inputs,
            # decoder_start_token_id = trained_tokenizer.eos_token_id,
            num_beams=8,
            do_sample=False,
            max_length=max_target_length
        )
    
    # print(f"generated original token: {outputs}")

    # 'trained_tokenizer.decode()'를 사용하여 모델이 생성한 토큰 ID 시퀀스('outputs[0]')를 사람이 읽을 수 있는 문자열로 변환합니다. 'skip_special_tokens=True'는 <pad>, <eos> 같은 특수 토큰을 결과에서 제외합니다.
    result_gloss = trained_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 'print' 함수와 f-string을 사용하여 최종적으로 생성된 gloss 문자열을 출력합니다.
    print(f"출력 Gloss: {result_gloss}")
    # 'split()' 메소드를 사용하여 생성된 gloss 문자열을 공백 기준으로 나누어 리스트 형태로 출력합니다.
    print(f"출력 Gloss (리스트): {result_gloss.split()}")
    return result_gloss.split()


# 이 스크립트 파일이 직접 실행될 때만 'main()' 함수를 호출하도록 하는 파이썬의 표준적인 구문입니다.
# 다른 스크립트에서 이 파일을 모듈로 임포트할 경우에는 'main()' 함수가 자동으로 실행되지 않습니다.
if __name__ == '__main__':
    # 'main' 함수를 호출하여 전체 프로세스를 시작합니다.
    inference("사춘기 때 아이에게 일어나는 변화를 잘 이해하고 지나가는 것이 필요합니다.")