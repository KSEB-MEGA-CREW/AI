import os
import shutil
import datetime

# --- 설정 부분 (사용자 직접 수정) ---

# 1. 검색을 시작할 외장 드라이브 경로를 지정하세요.
source_drive = "D:\\"

# 2. 최종 압축 파일을 저장할 로컬 경로를 지정하세요. (C드라이브 예시)
# 아래 경로에 최종 zip파일이 저장됩니다. 폴더는 자동으로 생성됩니다.
save_location = "C:\\압축파일모음"
local_dest_folder = os.path.join(save_location, "temp_files_for_zip")

# 3. 생성될 압축 파일의 이름을 지정하세요. (날짜와 시간이 자동으로 추가됩니다)
zip_file_base_name = "영상_JSON_압축파일"


# --- 코드 실행 부분 (수정 필요 없음) ---

def find_and_zip_pairs(source_path, dest_path, zip_name):
    """
    지정된 경로에서 json/mp4 파일 쌍을 찾아 로컬 경로에 복사하고 압축합니다.
    """
    # 1. 임시 저장 폴더 생성 (이미 있다면 비우고 다시 생성)
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    os.makedirs(dest_path)
    print(f"임시 폴더 생성: {dest_path}")

    found_files_count = 0

    # 2. 소스 드라이브의 모든 폴더와 파일을 순회
    print(f"'{source_path}'에서 파일 검색을 시작합니다...")
    for root, _, files in os.walk(source_path):
        mp4_basenames = {
            os.path.splitext(f)[0]
            for f in files
            if f.lower().endswith('.mp4') and not (f.lower().endswith('l.mp4') or f.lower().endswith('r.mp4'))
        }

        for filename in files:
            if filename.lower().endswith('.json'):
                json_basename = os.path.splitext(filename)[0]

                if json_basename in mp4_basenames:
                    json_full_path = os.path.join(root, filename)
                    mp4_full_path = os.path.join(root, json_basename + ".mp4")

                    print(f"  -> 쌍 발견: {filename}, {json_basename + '.mp4'}")

                    try:
                        shutil.copy2(json_full_path, dest_path)
                        shutil.copy2(mp4_full_path, dest_path)
                        found_files_count += 1
                    except Exception as e:
                        print(f"  !! 파일 복사 중 오류 발생: {e}")

    if found_files_count == 0:
        print("\n조건에 맞는 파일 쌍을 찾지 못했습니다.")
        shutil.rmtree(dest_path)
        return

    print(f"\n총 {found_files_count}개의 파일 쌍을 찾았으며, 복사를 완료했습니다.")

    # 4. 임시 폴더를 zip으로 압축
    print("파일 압축을 시작합니다...")
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_zip_name = f"{zip_name}_{timestamp}"

        # 최종 압축 파일은 임시 폴더의 상위 디렉토리에 저장
        archive_path = shutil.make_archive(
            os.path.join(os.path.dirname(dest_path), final_zip_name),
            'zip',
            dest_path
        )
        print(f"\n압축 완료! 파일이 다음 위치에 저장되었습니다: {archive_path}")
    except Exception as e:
        print(f"\n!! 파일 압축 중 오류 발생: {e}")
    finally:
        # 5. 임시 폴더 삭제
        print("임시 파일을 정리합니다...")
        shutil.rmtree(dest_path)
        print("작업이 모두 완료되었습니다.")


if __name__ == "__main__":
    if not os.path.exists(source_drive):
        print(f"오류: 소스 드라이브 '{source_drive}'를 찾을 수 없습니다. 경로를 확인해주세요.")
    else:
        find_and_zip_pairs(source_drive, local_dest_folder, zip_file_base_name)