import cv2
import mediapipe as mp
import xgboost as xgb
import numpy as np
import json
import os
from flask import Flask, render_template, send_from_directory, request, jsonify, Response, session

# GLOG_minloglevel 환경 변수 설정
os.environ['GLOG_minloglevel'] = '2'

app = Flask(__name__, static_url_path='/models', static_folder='static/models')
app.secret_key = 'your_secret_key'  # 세션 사용을 위한 secret key 설정 (보안상 강력한 키로 변경 필요)

# 정적 파일 서빙 라우트
@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('static/images', filename)

@app.route('/static/sounds/<path:filename>')
def serve_sounds(filename):
    return send_from_directory('static/sounds', filename)

# MediaPipe 및 XGBoost 모델 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# XGBoost 모델 및 특징 로드
try:
    model = xgb.XGBRegressor()
    # 모델 파일 경로를 현재 스크립트 파일 기준으로 설정
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, 'expression_similarity_model.json')
    feature_cols_path = os.path.join(script_dir, 'feature_cols.json')

    model.load_model(model_path)
    with open(feature_cols_path, 'r') as f:
        feature_cols = json.load(f)
    print("XGBoost 모델 및 특징 파일 로드 완료.")
except Exception as e:
    print(f"XGBoost 모델 또는 특징 파일 로드 실패: {e}")
    model = None
    feature_cols = []

# AU 계산을 위한 랜드마크 인덱스 (FACS 기준, MediaPipe 랜드마크 기반)
AU_LANDMARKS = {
    'AU01': [336, 296],  # 이마 안쪽 (inner brow raiser)
    'AU02': [334, 298],  # 이마 바깥쪽 (outer brow raiser)
    'AU04': [9, 8],     # 눈썹 내림 (brow furrower)
    'AU06': [205, 206],  # 광대 (cheek raiser)
    'AU12': [308, 78],   # 입꼬리 (lip corner puller)
    'AU25': [13, 14]    # 입술 개방 (lips part)
}

def calculate_au(landmarks):
    """MediaPipe 랜드마크 기반 AU 계산 (거리 기반)"""
    au_dict = {}

    if not isinstance(landmarks, np.ndarray) or not landmarks.any():
        for col in feature_cols:
            au_dict[col] = 0.0
        return au_dict

    # 기본 AU 계산 (랜드마크 간 유클리드 거리)
    for au, indices in AU_LANDMARKS.items():
        if indices[0] < len(landmarks) and indices[1] < len(landmarks):
            p1 = landmarks[indices[0]]
            p2 = landmarks[indices[1]]
            au_dict[au] = np.linalg.norm(p1 - p2)
        else:
            au_dict[au] = 0.0  # 랜드마크 인덱스가 유효하지 않으면 0

    # 가중치 컬럼 생성 (임시 0.8 곱함) - 이 부분은 모델 학습 방식에 따라 조정 필요
    for au_key in AU_LANDMARKS.keys():
        au_dict[f"{au_key}_w"] = au_dict.get(au_key, 0.0) * 0.8

    # 모델의 feature_cols에 있는 모든 컬럼을 포함하도록 final_au_features 구성
    final_au_features = {}
    for col in feature_cols:
        final_au_features[col] = au_dict.get(col, 0.0)  # au_dict에 없는 컬럼은 0

    return final_au_features

# 웹캠 객체를 전역 변수로 관리
global_cap = None

def generate_frames():
    global global_cap
    if global_cap is None or not global_cap.isOpened():
        global_cap = cv2.VideoCapture(0)
        if not global_cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            return

    frame_count = 0
    last_score = 0.0
    au_display_data = {au: 0.0 for au in AU_LANDMARKS.keys()}

    while global_cap.isOpened():
        success, frame = global_cap.read()
        if not success:
            break

        # 프레임 좌우 반전 (거울 효과)
        frame = cv2.flip(frame, 1)

        try:
            # MediaPipe 처리
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                # 랜드마크 좌표 추출
                landmarks = np.array([
                    (lm.x * frame.shape[1], lm.y * frame.shape[0])
                    for lm in results.multi_face_landmarks[0].landmark
                ], dtype=np.float32)

                # 5프레임마다 AU 계산 및 예측
                if frame_count % 5 == 0:
                    current_au_features = calculate_au(landmarks)
                    au_display_data = {k: v for k, v in current_au_features.items() if k.replace('_w', '') in AU_LANDMARKS.keys()}

                    if model is not None and feature_cols:
                        user_X = [[current_au_features.get(col, 0.0) for col in feature_cols]]
                        last_score = model.predict(user_X)[0]
                    else:
                        last_score = 0.0
            else:
                # 얼굴이 감지되지 않으면 AU와 점수 초기화
                au_display_data = {au: 0.0 for au in AU_LANDMARKS.keys()}
                last_score = 0.0

            # 텍스트 오버레이 그리기 및 반전 처리
            temp_text_overlay = np.zeros_like(frame)

            y_offset = 30
            for i, (au_key, au_value) in enumerate(au_display_data.items()):
                display_au_key = au_key.replace('_w', '')
                cv2.putText(temp_text_overlay, f"{display_au_key}: {au_value:.2f}", (10, y_offset + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(temp_text_overlay, f"Score: {last_score:.2f}", (10, y_offset + len(au_display_data) * 30 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            flipped_text_overlay = cv2.flip(temp_text_overlay, 1)

            gray_flipped_text = cv2.cvtColor(flipped_text_overlay, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_flipped_text, 1, 255, cv2.THRESH_BINARY)

            mask_inv = cv2.bitwise_not(mask)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_inv_rgb = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

            frame = cv2.bitwise_and(frame, mask_inv_rgb)
            frame = cv2.add(frame, cv2.bitwise_and(flipped_text_overlay, mask_rgb))

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            frame_count += 1

        except Exception as e:
            print(f"스트리밍 중 오류 발생: {str(e)}")
            continue

    global_cap.release()

# 페이지 라우팅
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/game_follow')
def game_follow():
    return render_template('game_follow.html')  # 표정 따라하기 모드

@app.route('/game_feedback')
def game_feedback():
    return render_template('game_feedback.html')

@app.route('/rehab_mode')
def rehab_mode():
    return render_template('rehab_mode.html')

@app.route('/focus')
def focus():
    return render_template('focus.html')

@app.route('/complex')
def complex():
    return render_template('complex.html')

@app.route('/complex_fit')
def complex_fit():
    teacher = request.args.get('teacher')
    return render_template('complex_fit.html', teacher=teacher)

@app.route('/focus_fit')
def focus_fit():
    return render_template('focus_fit.html')

@app.route('/feedback')
def feedback():
    # 세션에서 캡처된 이미지 데이터 가져오기
    try:
        captured_images_json = session.pop('capturedImages', '[]')
        captured_images = json.loads(captured_images_json)
        selected_teacher = session.pop('selectedTeacher', 'default_teacher')
        mode = session.pop('mode', 'default_mode')  # 모드 값 가져오기
    except RuntimeError:  # 세션 컨텍스트 밖에서 접근 시 발생할 수 있는 에러 처리
        captured_images = []
        selected_teacher = 'default_teacher'
        mode = 'default_mode'

    return render_template('feedback.html',
                           captured_images=captured_images,
                           selected_teacher=selected_teacher,
                           mode=mode)  # 모드 값 전달

@app.route('/game_mode')
def game_mode():
    return render_template('game_mode.html')

@app.route('/game_emotion')
def game_emotion():
    return render_template('game_emotion.html')

@app.route('/play_tower_defense')
def play_tower_page():
    # URL에서 'mode' 파라미터 가져오기, 없으면 'default_mode' 사용
    selected_mode = request.args.get('mode', 'default_mode')

    # 'tower_defense_game.html' 템플릿을 렌더링하면서 current_game_mode 변수로 mode 값 전달
    return render_template('tower_defense_game.html', current_game_mode=selected_mode)


# 비디오 피드를 위한 라우트 (complex_fit.html에서 사용)
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # 웹캠이 열려있다면 닫기 (서버 재시작 시 필요)
    if global_cap is not None and global_cap.isOpened():
        global_cap.release()
    app.run(debug=True, host='0.0.0.0', port=5000)