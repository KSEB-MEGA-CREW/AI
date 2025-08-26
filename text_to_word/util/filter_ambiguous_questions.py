
import json
import os

def filter_ambiguous_questions(input_file_path, final_dataset_path, ambiguous_questions_path):
    """
    Filters out only ambiguous interrogative sentences.
    An ambiguous question is one that is a question in Korean but has no corresponding question-word gloss.

    모호한 의문문만 필터링하여 제거합니다.
    모호한 의문문이란, 한국어로는 의문문이지만 해당하는 의문사 글로스가 없는 문장을 의미합니다.
    """
    # List of question-word glosses found in KSL
    # 한국수어의 의문사 글로스 목록
    QUESTION_GLOSSES = {
        '왜1', '무엇1', '무엇2', '어디1', '누구1', '언제1', '어떻게1', '방법1', '얼마1', '몇1' '어디3', '어떻게2'
    }

    final_dataset = []
    ambiguous_questions = []

    # Common Korean question endings
    # 일반적인 한국어 의문문 어미
    question_endings = ('?', '까', '까요', '나요', '가요', '뭔가요', '을까요')

    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                korean_text = data.get("koreanText", "").strip()
                gloss_ids = set(data.get("gloss_id", []))

                is_question = korean_text.endswith(question_endings)

                if not is_question:
                    # It's a statement, so keep it.
                    # 평서문이므로 유지합니다.
                    final_dataset.append(data)
                    continue

                # The sentence is a question. Check for question glosses.
                # 문장이 의문문입니다. 의문사 글로스가 있는지 확인합니다.
                if not QUESTION_GLOSSES.intersection(gloss_ids):
                    # This is an ambiguous question (e.g., Yes/No question). Remove it.
                    # 모호한 의문문(예: 예/아니오 질문)이므로 제거합니다.
                    ambiguous_questions.append(data)
                else:
                    # This question has an explicit question word. Keep it.
                    # 명시적인 의문사 단어가 있는 질문이므로 유지합니다.
                    final_dataset.append(data)

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from line.")

    # Save the final, filtered dataset
    # 최종 필터링된 데이터셋 저장
    with open(final_dataset_path, 'w', encoding='utf-8') as f:
        for item in final_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Save the removed ambiguous questions for inspection
    # 제거된 모호한 의문문들을 분석용으로 저장
    with open(ambiguous_questions_path, 'w', encoding='utf-8') as f:
        for item in ambiguous_questions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("정교한 데이터 필터링 완료 (Sophisticated data filtering complete)")
    print(f"총 {len(final_dataset) + len(ambiguous_questions)}개 데이터 처리 (Total {len(final_dataset) + len(ambiguous_questions)} items processed)")
    print(f"  - 최종 학습 데이터셋 (Final training set): {len(final_dataset)}개 -> {final_dataset_path}")
    print(f"  - 제거된 모호한 의문문 (Removed ambiguous questions): {len(ambiguous_questions)}개 -> {ambiguous_questions_path}")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Use 'final_data_replaced.jsonl' which has rare words processed
    # 희귀 단어가 처리된 'final_data_replaced.jsonl' 사용
    input_path = os.path.join(os.path.dirname(current_dir), 'preprocessed_data', 'cleaned_data.jsonl')
    
    # Define output paths
    # 출력 경로 정의
    final_dataset_path = os.path.join(os.path.dirname(current_dir), 'preprocessed_data', 'dataset_final_filtered.jsonl')
    ambiguous_questions_path = os.path.join(os.path.dirname(current_dir), 'preprocessed_data', 'dataset_ambiguous_questions.jsonl')

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        print("Please run process_rare_words.py first to generate it.")
    else:
        filter_ambiguous_questions(input_path, final_dataset_path, ambiguous_questions_path)
