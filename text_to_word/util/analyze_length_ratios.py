
import json
import numpy as np
import os
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_ratios(file_path, output_dir):
    """
    Analyzes the length ratio of Korean text to gloss IDs and suggests an optimal threshold.
    한국어 텍스트와 글로스 ID의 길이 비율을 분석하고 최적의 임계값을 제안합니다.
    """
    ratios = []
    total_lines = 0
    parse_errors = 0
    zero_len_count = 0

    print(f"파일 분석 중: {file_path}")
    # Analyzing file

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            try:
                data = json.loads(line)
                korean_text = data.get("koreanText", "")
                gloss_ids = data.get("gloss_id", [])

                korean_text = re.sub(r'\s+', ' ', korean_text).strip()
                korean_len = len(korean_text.split())
                gloss_len = len(gloss_ids)

                if korean_len == 0 or gloss_len == 0:
                    zero_len_count += 1
                    continue

                ratio = max(korean_len, gloss_len) / min(korean_len, gloss_len)
                ratios.append(ratio)

            except json.JSONDecodeError:
                parse_errors += 1
                continue

    print(f"총 라인 수: {total_lines}")
    # Total lines processed
    print(f"파싱 오류 라인 수: {parse_errors}")
    # Lines with parsing errors
    print(f"길이가 0인 데이터 수: {zero_len_count}")
    # Lines with zero-length text or gloss
    print(f"분석 대상 데이터 수: {len(ratios)}")
    # Number of valid data pairs for ratio analysis
    print("-" * 30)

    if not ratios:
        print("분석할 데이터가 없습니다.")
        # No valid data to analyze
        return

    # --- 분석 (Analysis) ---
    print("길이 비율 분포 분석 (Length Ratio Distribution Analysis)")
    print("-" * 30)

    # 1. 백분위수 (Percentiles)
    percentiles = [90, 95, 98, 99, 99.5, 99.9]
    p_values = np.percentile(ratios, percentiles)
    print("백분위수 (Percentiles):")
    # Meaning: 95% of the data has a length ratio of {p_values[1]:.2f} or less.
    for p, val in zip(percentiles, p_values):
        print(f"  - 상위 {100-p}% 지점 (Top {100-p}% point): {val:.2f}")
    print(f"  (의미: 데이터의 95%는 길이 비율이 {p_values[1]:.2f} 이하입니다.)")
    print("-" * 30)

    # 2. 정수 임계값에 따른 데이터 손실 분석 (Data Loss at Integer Thresholds)
    print("정수 임계값에 따른 데이터 손실 분석 (Data Loss at Integer Thresholds):")
    thresholds_to_check = np.arange(1.0, 3.0, 0.2)
    total_valid_data = len(ratios)
    for t in thresholds_to_check:
        removed = sum(1 for r in ratios if r > t)
        percentage_removed = (removed / total_valid_data) * 100
        print(f"  - 임계값 > {t}: 제거될 데이터 {removed}개 ({percentage_removed:.2f}%)")
        # Data to be removed for threshold > {t}: {removed} items ({percentage_removed:.2f}%)
    print("-" * 30)

    # 3. 시각화 (Visualization)
    plt.figure(figsize=(12, 6))
    sns.histplot(ratios, bins=100, kde=True)
    plt.title('Length Ratio Distribution (Korean vs. Gloss)')
    plt.xlabel('Length Ratio (max/min)')
    plt.ylabel('Frequency')
    plt.xlim(0, 10) # 비율이 10 이상인 데이터는 제외하고 표시 (Exclude data with ratio > 10 for better visualization)
    
    # 백분위수 라인 추가 (Add percentile lines)
    for p, val in zip(percentiles, p_values):
        if val <= 10:
            plt.axvline(val, color='red', linestyle='--', linewidth=1)
            plt.text(val + 0.1, plt.ylim()[1] * 0.9, f'{p}%: {val:.2f}', rotation=90, verticalalignment='center', color='red')

    plot_path = os.path.join(output_dir, 'length_ratio_distribution.png')
    plt.savefig(plot_path)
    print(f"분포 그래프가 '{plot_path}'에 저장되었습니다.")
    # Distribution graph saved to '{plot_path}'
    print("-" * 30)


    # 4. 추천 (Recommendation)
    p99_threshold = p_values[3] # 99th percentile
    print("추천 (Recommendation):")
    # The current threshold of 4 seems reasonable. In this case, about {sum(1 for r in ratios if r > 4) / total_valid_data * 100:.2f}% of the data will be removed.
    print(f"현재 임계값 4는 약 {sum(1 for r in ratios if r > 4) / total_valid_data * 100:.2f}%의 데이터를 제거하며, 대부분의 데이터를 유지하면서 일부 이상치를 제거하는 합리적인 선택으로 보입니다.")
    # If you want to preserve more data, you could consider setting the threshold to the 99th percentile, which is {p99_threshold:.2f} (approximately {round(p99_threshold, 1)}).
    print(f"만약 더 많은 데이터를 보존하고 싶다면, 99백분위수에 해당하는 임계값인 {p99_threshold:.2f} (약 {round(p99_threshold, 1)}) 정도로 설정하는 것을 고려해볼 수 있습니다.")
    # However, be cautious about setting the threshold too high, as removing extreme outliers can be beneficial for model stability.
    print("하지만 극단적인 이상치를 제거하는 것이 모델 안정성에 도움이 될 수 있으므로, 임계값을 너무 높게 설정하는 것은 주의해야 합니다.")


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    input_path = os.path.join(project_root, 'preprocessed_data', 'processed_data.jsonl')
    output_dir = os.path.join(project_root, 'plots') # Save plots to the 'plots' directory

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(input_path):
        print(f"오류: 입력 파일을 찾을 수 없습니다. 경로: {input_path}")
        # Error: Input file not found at {input_path}
    else:
        analyze_ratios(input_path, output_dir)
