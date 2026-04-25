# Background Cropper

Streamlit 기반 로컬 이미지 처리 앱입니다.

현재 기능:

- 단일 이미지 업로드
- `rembg`로 배경 제거
- 투명 영역 기준 오브젝트 bbox 계산
- 투명 공백 전체 제거 후 PNG로 crop
- 가로/세로 출력 크기 지정
- 비율 잠금 토글
- 강제 늘리기 또는 비율 유지 중앙 배치 리사이즈
- 다운로드 또는 `outputs/` 폴더 저장

## 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

`rembg`는 첫 실행 시 모델 파일을 다운로드할 수 있습니다.
