# Background Cropper

Streamlit 기반 로컬 이미지 처리 앱입니다.

현재 기능:

- 단일 또는 다중 이미지 업로드
- 스프라이트 시트 배경 제거 후 연결되지 않은 요소별 자동 분리
- `rembg`로 배경 제거
- 투명 영역 기준 오브젝트 bbox 계산
- 투명 공백 전체 제거 후 PNG로 crop
- 화면의 이미지별 카드에서 배경 제거/마스크/출력 크기 옵션 조정
- 가로/세로 출력 크기 지정
- 비율 잠금 토글
- 강제 늘리기 또는 비율 유지 중앙 배치 리사이즈
- 선택 이미지 다운로드
- 선택 이미지 저장 또는 전체 이미지 일괄 `outputs/` 폴더 저장

## 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

`rembg`는 첫 실행 시 모델 파일을 다운로드할 수 있습니다.
