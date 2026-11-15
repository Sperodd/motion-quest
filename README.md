# Pose Lab — 브라우저 실시간 동작 분류 테스트

클라이언트는 MediaPipe Lite의 공통 12관절 XYZ만 전송합니다. 서버는 같은 XYZ에서 **정규화된 위치 36개 + 각도 특징 18개**를 계산하여 54채널 TCN으로 분류합니다.

```text
웹캠 → 브라우저 MediaPipe Lite → 공통 12관절 XYZ JSON / WebSocket
 → 서버: 최근 3초 XYZ를 30 Hz로 보간
 → [몸 기준 위치 36 | 각도 18] = [90,54]
 → 모델 내부 채널별 표준화 → TCN → 5클래스 점수 → 브라우저
```

영상은 서버로 보내지 않습니다. 화면 미러링은 좌표를 바꾸지 않습니다. 버퍼는 연결마다 독립적입니다.

## 위치+각도 모델로 전환 — 재학습 필수

1. `ui_prmd_tcn_colab.ipynb`를 Colab에서 처음부터 실행합니다. 원본 Positions의 XYZ에서 두 특징을 모두 계산하며 UI-PRMD Angles 파일은 사용하지 않습니다.
2. 결과는 `/content/ui_prmd_tcn_output`과 `ui_prmd_tcn_output.zip`에 저장됩니다. Drive 복사 이름도 `ui_prmd_tcn_output.zip`입니다. 같은 이름의 이전 결과가 있다면 비교를 위해 미리 보관하세요.
3. ZIP을 프로젝트 루트에 풀어 `ui_prmd_tcn_output/metadata.json`, `preprocess.py`, `pose_tcn.torchscript.pt`가 함께 있게 하고 서버를 재시작합니다.
4. 화면의 **위치 + 각도 모델 · 54개 특징**, 또는 `/api/config`의 `feature_schema: positions_angles_v1`을 확인합니다.

서버 기본 경로는 `ui_prmd_tcn_output`이며 이전 `ui_prmd_tcn_angles_output`을 자동 선택하지 않습니다. `POSE_MODEL_DIR`가 설정돼 있다면 해당 값이 우선합니다. 새 모델이 아직 없을 때 이전 각도 모델을 명시적으로 실행하려면 PowerShell에서 `$env:POSE_MODEL_DIR = 'ui_prmd_tcn_angles_output'`을 지정합니다. 새 통합 모델 설치 후에는 `$env:POSE_MODEL_DIR = 'ui_prmd_tcn_output'`으로 바꾸세요.

기존 18/36채널 가중치는 통합 54채널 입력에 사용할 수 없습니다. 기존 학습 파일은 변경하지 않았으며 새로운 가중치는 재학습 후 생성됩니다. 파일 일부만 섞어 교체하면 서버의 입력 계약 검사가 거부합니다.

### 특징 순서와 공통 전처리

`pose_features.py`가 학습·서버 공통 소스입니다.

- **0~35 위치:** 프레임별 골반 중심 이동 → 창 중앙값 몸통 길이로 스케일 정규화 → 창 평균 좌우 골반 방향과 수직축으로 고정한 몸 좌표계. 12관절 × [lateral, up, depth] 순서입니다. 절대 위치/체격을 제거하고 상대적인 발 배치와 몸통 기울기를 남깁니다.
- **36~53 각도:** 좌우 무릎·골반·팔꿈치·어깨(8), 허벅지/종아리 사이(2), 몸통과 수직축(1), 좌우 허벅지/종아리와 수직축(4), 좌우 무릎·골반·허벅지 기울기 차이(3).
- **공식:** A–B–C 내각은 `acos(clip(dot(unit(A-B), unit(C-B)), -1, 1))/π`. 0~1은 0~180°입니다. 위치는 부호를 가지며 0~1로 제한하지 않습니다.
- **표준화:** train에서 54개 채널 각각의 평균/표준편차를 계산하고 모델 내부에서 적용합니다. 위치와 각도의 단위 차이를 처리합니다.
- **좌우 증강:** 두 블록을 함께 교환하고, 위치 lateral 성분만 부호 반전합니다.

Vicon up=+Z, MediaPipe up=-Y입니다. 카메라 기울기는 별도 보정 대상입니다. 위치는 공간 배치, 각도는 굽힘/비대칭을 직접 표현하지만 각도도 XYZ에서 파생되므로 새 센서 정보가 추가되는 것은 아닙니다. 실제 MediaPipe에서의 향상 여부는 위치 전용/각도 전용/통합 모델을 같은 영상으로 비교해야 합니다.

공통 소스를 수정한 후 `python scripts/sync_notebook_features.py`로 노트북에 삽입하세요. Colab에는 ipynb만 올리면 됩니다. 저장 소스 해시·특징 순서·단위·차원을 서버에서 검사합니다.

## 실행 (프로젝트 루트, PowerShell)

Python 3.11 이상에서:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --ws-max-size 16384 --ws-max-queue 4
```

브라우저에서 http://localhost:8000 을 열고 **카메라 시작**을 누릅니다. 전신(특히 양쪽 발목)이 보이도록 촬영하고 같은 동작을 3초 이상 반복합니다. 첫 결과까지 약 3초가 필요하고, 동작을 바꿨을 때도 이전 동작이 창에서 빠지는 시간이 필요합니다.

JS/WASM과 Lite 모델은 CDN에서 다운로드하므로 최초 실행에 인터넷 연결이 필요합니다. 카메라는 localhost 또는 HTTPS에서 사용할 수 있습니다. LAN IP의 일반 HTTP로 휴대폰에서 접속하는 방식은 지원하지 않습니다. 기본 서버는 로컬 테스트 전용이며 공개 배포를 위한 인증/접속 제한은 구현하지 않았습니다.

모델을 바꾸려면 `POSE_MODEL_DIR` 환경변수를 학습 출력 폴더로 지정합니다. `metadata.json`, `preprocess.py`, `pose_tcn.torchscript.pt`가 함께 필요합니다. 모델 내부에 표준화가 들어 있으므로 서버는 평균/표준편차를 다시 적용하지 않습니다. PyTorch 로드 실패 시 학습 환경 버전은 `metadata.json`에서 확인하세요.

## JSON 계약 v1

`GET /api/config`가 관절 순서, MediaPipe 인덱스, 클래스 순서, FPS와 창 길이를 반환합니다. 12개 관절의 순서는 **좌/우 어깨 → 팔꿈치 → 손목 → 골반 → 무릎 → 발목**입니다. 이미지의 normalized landmarks가 아니라 **worldLandmarks (XYZ, meters)** 를 보냅니다. visibility는 같은 관절의 이미지 landmark 값을 사용합니다.

클라이언트 → `/ws/pose` (joints는 아래 형식의 객체 12개):

```json
{
  "version": 1,
  "type": "frame",
  "coordinate_system": "mediapipe_world",
  "timestamp_ms": 1250.5,
  "joints": [{"x": -0.2, "y": -0.6, "z": 0.01, "visibility": 0.99}]
}
```

위 예시는 관절 한 개만 축약한 것이며 실제 요청에는 12개가 필요합니다. `timestamp_ms`는 브라우저 `performance.now()` 기준의 단조 증가 밀리초입니다. 프레임마다 한 응답을 받습니다. 브라우저는 미응답 요청을 하나만 유지하여 지연된 프레임의 누적을 방지합니다. 서버 추론은 스레드로 실행하여 이벤트 루프를 막지 않습니다.

서버 → 클라이언트 예시(예시 수치이며 측정 결과가 아님):

```json
{
  "status": "prediction",
  "class_id": "m01",
  "label": "M01 Deep Squat",
  "score": 0.8,
  "probabilities": [0.8, 0.05, 0.05, 0.05, 0.05],
  "inference_ms": 4.2,
  "timestamp_ms": 4500,
  "progress": 1
}
```

다른 상태는 `warming_up`(수집), `collecting`(다음 추론까지 수집), `tracking_lost`(명시적 reset 응답), `invalid_input`(잘못된 요청)입니다. 현재 실험 모드는 visibility와 추적 간격으로 프레임을 거르거나 버퍼를 초기화하지 않습니다. 좌표가 아예 없으면 브라우저는 해당 프레임만 건너뛰고, 다시 들어온 좌표 사이의 공백은 서버가 선형 보간합니다. 긴 공백도 보간하므로 실제로 관측하지 않은 움직임이 분류에 포함될 수 있습니다. 잘못된 숫자/관절 수/시간 순서는 거부하되 버퍼는 보존합니다. 중지/탭 숨김/연결 종료 시 카메라와 기존 결과를 정리합니다.

통합 모델 경로에서는 서버가 12관절을 33-landmark 형식으로 어댑트한 뒤 **학습 산출물 `preprocess.py`의 mediapipe_json_window → window_features를 호출**합니다. XYZ 시간 보간 후 위치 정규화와 각도 계산을 모두 수행합니다. 각도 자체를 보간하지 않습니다. 학습은 Vicon 100→30 Hz anti-alias resampling 후 같은 통합 특징 함수를 사용합니다. 기존 위치 모델은 종전 normalize_window 경로를 유지합니다. 저장된 코드이므로 신뢰하는 자신의 학습 산출물만 사용하세요.

## 테스트 및 한계

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node --check web/app.js
# 로컬 원본 데이터 전체의 각도 입력 검사 (pandas/scipy가 설치되어 있을 때)
python scripts/validate_angle_data.py
```

테스트는 기존 저장 모델 회귀 검사 및 새 54채널 TCN의 학습 연산/저장/로드, WebSocket 왕복, XYZ→위치+각도 입력 일치, 0/90/180도 공식, 단위/좌표계 변환, 좌우 반전, 잘못된 입력과 메타데이터 계약을 확인합니다. 통합 모델 테스트 가중치는 임시로 생성하며 실제 재학습 결과나 정확도 검증이 아닙니다.

이전 모델의 Vicon 정확도와 별개로 새 통합 모델 및 MediaPipe 영상 정확도는 별도로 측정해야 합니다. 이 모델은 정지/배경/unknown을 학습하지 않아 어떤 입력도 5개 클래스 중 하나로 선택합니다. 화면의 점수는 softmax이며 올바른 운동 자세를 평가하는 점수가 아닙니다. 임의 임계값이나 평활화로 결과를 가리지 않고 원 모델 출력을 표시합니다.

## 이후 Godot 연결

분류 프로토콜을 UI와 분리했습니다. 추후 Godot의 WebSocket 클라이언트가 같은 `/ws/pose` 계약을 사용하거나, 브라우저가 추정한 좌표를 넘겨주는 중계 계층을 추가할 수 있습니다. 현재는 브라우저가 자기 연결의 결과를 받으며, 별도 Godot 연결에 결과를 방송하는 기능은 아직 없습니다. 관절 추정 기능 자체를 Godot에 넣는 작업도 별도입니다.

참고: [Google MediaPipe Web 가이드](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker?tab=Web).
