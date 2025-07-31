import os
import numpy as np
from collections import Counter

DATA_PATH = r"C:\SoftwareEdu2025\project\Hand_Sound\KCH\signtotext\output_npy\cleaned_npy"
frame_lengths = []

for root, dirs, files in os.walk(DATA_PATH):
    for fname in files:
        if fname.endswith(".npy"):
            arr = np.load(os.path.join(root, fname))
            frame_lengths.append(arr.shape[0])

print("프레임 길이 분포:", Counter(frame_lengths))
print("최소프레임:", min(frame_lengths), "최대프레임:", max(frame_lengths))
print("중앙값:", np.median(frame_lengths), "평균:", np.mean(frame_lengths))