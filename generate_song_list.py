#!/usr/bin/env python3
import os
import json
from pathlib import Path

# 경로 설정
BASE_DIR = "./demo_src/POP909_select_inst_8bar_cropped/valid"
OUTPUT_FILE = "./demo_src/songs_with_segments.json"

def generate_song_list(base_dir):
    """
    주어진 디렉토리에서 폴더 목록과 각 폴더 내의 MIDI 파일 목록을 가져옵니다.
    폴더명을 키로, 해당 폴더 내의 MIDI 파일 목록을 값으로 하는 딕셔너리를 반환합니다.
    """
    result = {}
    
    # 베이스 디렉토리가 존재하는지 확인
    if not os.path.exists(base_dir):
        print(f"오류: 디렉토리가 존재하지 않습니다: {base_dir}")
        return result
    
    # 모든 하위 폴더 탐색
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    folders.sort()  # 이름순으로 정렬
    
    for folder in folders:
        if folder == 'segment_meta':
            continue
        folder_path = os.path.join(base_dir, folder)
        # MIDI 파일 목록 가져오기 (.mid 확장자 파일)
        midi_files = [f for f in os.listdir(folder_path) if f.endswith('.mid')]
        
        # 숫자를 기준으로 정렬하기 위한 키 함수 정의
        def get_segment_number(filename):
            # seg_X_Y.mid 형식에서 X를 추출하여 정수로 변환
            start_num = int(filename.split('_')[1])
            return start_num
            
        # 숫자 기준으로 정렬
        midi_files.sort(key=get_segment_number)
        
        # 확장자 제거
        midi_files = [f.split('.')[0] for f in midi_files]
        
        # 결과 딕셔너리에 추가
        result[folder] = midi_files
    
    return result

def main():
    print(f"디렉토리 탐색 중: {BASE_DIR}")
    
    # 폴더 목록과 MIDI 파일 목록 가져오기
    song_list = generate_song_list(BASE_DIR)
    
    if not song_list:
        print("오류: 데이터를 가져올 수 없습니다.")
        return
    
    # JSON 파일로 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(song_list, f, ensure_ascii=False, indent=4)
    
    print(f"노래 목록이 저장되었습니다: {OUTPUT_FILE}")
    print(f"총 {len(song_list)} 개의 폴더가 발견되었습니다.")

if __name__ == "__main__":
    main()
