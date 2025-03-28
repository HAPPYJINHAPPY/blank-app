import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

# Initialize models
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def get_coord(landmark, model_type='pose', img_width=640, img_height=480):
    """Unified 3D coordinate processing (hand depth zero-padded)"""
    if model_type == 'pose':
        return [landmark.x * img_width, landmark.y * img_height, landmark.z * img_width]
    elif model_type == 'hands':
        return [landmark.x * img_width, landmark.y * img_height, 0]  # Zero-pad hand depth

def calculate_angle(a, b, c, plane='sagittal'):
    """Safe 3D angle calculation"""
    try:
        # Ensure 3D
        a = np.array(a)[:3].astype('float64')
        b = np.array(b)[:3].astype('float64')
        c = np.array(c)[:3].astype('float64')

        # Vector calculation
        ba = a - b
        bc = c - b

        # Plane projection
        if plane == 'sagittal':
            ba = np.array([0, ba[1], ba[2]])
            bc = np.array([0, bc[1], bc[2]])
        elif plane == 'frontal':
            ba = np.array([ba[0], 0, ba[2]])
            bc = np.array([bc[0], 0, bc[2]])
        elif plane == 'transverse':
            ba = ba[:2]
            bc = bc[:2]

        # Zero vector handling
        ba_norm = np.linalg.norm(ba)
        bc_norm = np.linalg.norm(bc)
        if ba_norm < 1e-6 or bc_norm < 1e-6:
            return 0.0

        # Angle calculation
        cosine = np.dot(ba, bc) / (ba_norm * bc_norm)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    except Exception as e:
        print(f"Angle calculation error: {str(e)}")
        return 0.0

def calculate_neck_flexion(nose, shoulder_mid, hip_mid):
    """Calculate neck flexion angle (deviation from neutral)"""
    try:
        # Convert to numpy arrays
        nose = np.array(nose)[:2]
        shoulder_mid = np.array(shoulder_mid)[:2]
        hip_mid = np.array(hip_mid)[:2]

        # Calculate torso axis
        torso_vector = hip_mid - shoulder_mid
        torso_angle = np.degrees(np.arctan2(torso_vector[1], torso_vector[0]))

        # Calculate head vector
        head_vector = nose - shoulder_mid
        head_angle = np.degrees(np.arctan2(head_vector[1], head_vector[0]))

        # Calculate deviation
        flexion_angle = head_angle - torso_angle

        # 规范化角度到 0-180 度范围
        if flexion_angle < 0:
            flexion_angle += 360
        if flexion_angle > 180:
            flexion_angle = 360 - flexion_angle

        # 转换为偏离中心位的角度
        flexion_angle = 180 - flexion_angle

        return flexion_angle
    except Exception as e:
        print(f"Neck flexion error: {str(e)}")
        return 0.0

def calculate_trunk_flexion(shoulder_mid, hip_mid, knee_mid):
    """Calculate trunk flexion angle (deviation from neutral)"""
    try:
        # Torso vector
        torso_vector = np.array(hip_mid) - np.array(shoulder_mid)
        torso_angle = np.degrees(np.arctan2(torso_vector[1], torso_vector[0]))

        # Leg vector
        leg_vector = np.array(knee_mid) - np.array(hip_mid)
        leg_angle = np.degrees(np.arctan2(leg_vector[1], leg_vector[0]))

        # Flexion calculation
        flexion_angle = leg_angle - torso_angle
        # 规范化角度到 0-180 度范围
        if flexion_angle < 0:
            flexion_angle += 360
        if flexion_angle > 180:
            flexion_angle = 360 - flexion_angle


        return flexion_angle
    except Exception as e:
        print(f"Trunk flexion error: {str(e)}")
        return 0.0

def process_image(image):
    H, W, _ = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Parallel processing
    pose_result = pose.process(img_rgb)
    hands_result = hands.process(img_rgb)

    metrics = {'angles': {}}

    if pose_result.pose_landmarks:
        def get_pose_pt(landmark):
            return get_coord(pose_result.pose_landmarks.landmark[landmark], 'pose', W, H)

        joints = {
            'left_side': {
                'shoulder': get_pose_pt(mp_pose.PoseLandmark.LEFT_SHOULDER),
                'elbow': get_pose_pt(mp_pose.PoseLandmark.LEFT_ELBOW),
                'wrist': get_pose_pt(mp_pose.PoseLandmark.LEFT_WRIST),
                'hip': get_pose_pt(mp_pose.PoseLandmark.LEFT_HIP),
                'knee': get_pose_pt(mp_pose.PoseLandmark.LEFT_KNEE)
            },
            'right_side': {
                'shoulder': get_pose_pt(mp_pose.PoseLandmark.RIGHT_SHOULDER),
                'elbow': get_pose_pt(mp_pose.PoseLandmark.RIGHT_ELBOW),
                'wrist': get_pose_pt(mp_pose.PoseLandmark.RIGHT_WRIST),
                'hip': get_pose_pt(mp_pose.PoseLandmark.RIGHT_HIP),
                'knee': get_pose_pt(mp_pose.PoseLandmark.RIGHT_KNEE)
            },
            'mid': {
                'shoulder_mid': [(get_pose_pt(mp_pose.PoseLandmark.LEFT_SHOULDER)[i] +
                                get_pose_pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)[i])/2 
                               for i in range(3)],
                'hip_mid': [(get_pose_pt(mp_pose.PoseLandmark.LEFT_HIP)[i] +
                           get_pose_pt(mp_pose.PoseLandmark.RIGHT_HIP)[i])/2 
                          for i in range(3)],
                'knee_mid': [(get_pose_pt(mp_pose.PoseLandmark.LEFT_KNEE)[i] +
                            get_pose_pt(mp_pose.PoseLandmark.RIGHT_KNEE)[i])/2 
                           for i in range(3)]
            },
            'nose': get_pose_pt(mp_pose.PoseLandmark.NOSE)
        }

        # Merge hand data
        if hands_result.multi_hand_landmarks:
            for hand in hands_result.multi_hand_landmarks:
                side = 'left_side' if hand.landmark[0].x < 0.5 else 'right_side'
                joints[side].update({
                    'hand_wrist': get_coord(hand.landmark[mp_hands.HandLandmark.WRIST], 'hands', W, H),
                    'index_mcp': get_coord(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP], 'hands', W, H),
                    'index_tip': get_coord(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP], 'hands', W, H)
                })

        try:
            # Calculate angles
            metrics['angles']['Neck Flexion'] = calculate_neck_flexion(
                joints['nose'], joints['mid']['shoulder_mid'], joints['mid']['hip_mid'])

            # Shoulder movements
            for side in ['left_side', 'right_side']:
                metrics['angles'][f'{side.capitalize()} Shoulder Elevation'] = calculate_angle(
                    joints[side]['hip'], joints[side]['shoulder'], joints[side]['elbow'], 'frontal')
                metrics['angles'][f'{side.capitalize()} Shoulder Flexion'] = calculate_angle(
                    joints[side]['hip'], joints[side]['shoulder'], joints[side]['elbow'], 'sagittal')

            # Elbow flexion
            for side in ['left_side', 'right_side']:
                metrics['angles'][f'{side.capitalize()} Elbow Flexion'] = calculate_angle(
                    joints[side]['shoulder'], joints[side]['elbow'], joints[side]['wrist'], 'sagittal')

            # Wrist movements
            for side in ['left_side', 'right_side']:
                if 'hand_wrist' in joints[side]:
                    metrics['angles'][f'{side.capitalize()} Wrist Dorsiflexion'] = calculate_angle(
                        joints[side]['elbow'], joints[side]['hand_wrist'], 
                        joints[side].get('index_tip', [0, 0, 0]), 'sagittal')
                    metrics['angles'][f'{side.capitalize()} Wrist Radial Deviation'] = calculate_angle(
                        joints[side]['index_mcp'], joints[side]['hand_wrist'],
                        joints[side].get('index_tip', [0, 0, 0]), 'frontal')

            # Trunk flexion
            metrics['angles']['Trunk Flexion'] = calculate_trunk_flexion(
                joints['mid']['shoulder_mid'], joints['mid']['hip_mid'], joints['mid']['knee_mid'])

            # Visualization
            draw_landmarks(image, joints)

        except KeyError as e:
            print(f"Missing keypoint: {str(e)}")

    return image, metrics

def draw_landmarks(image, joints):
    """Visualize joint connections"""
    colors = {
        'neck': (255, 200, 0),    # Gold
        'shoulder': (0, 255, 0),  # Green
        'elbow': (0, 255, 255),  # Cyan
        'wrist': (255, 0, 255)    # Magenta
    }

    # Neck line
    nose = tuple(map(int, joints['nose'][:2]))
    shoulder_mid = tuple(map(int, joints['mid']['shoulder_mid'][:2]))
    hip_mid = tuple(map(int, joints['mid']['hip_mid'][:2]))
    cv2.line(image, nose, shoulder_mid, colors['neck'], 2)
    cv2.line(image, shoulder_mid, hip_mid, colors['neck'], 2)

    # Upper limbs
    for side in ['left_side', 'right_side']:
        # Shoulder-elbow
        pt1 = tuple(map(int, joints[side]['shoulder'][:2]))
        pt2 = tuple(map(int, joints[side]['elbow'][:2]))
        cv2.line(image, pt1, pt2, colors['shoulder'], 2)

        # Elbow-wrist
        pt3 = tuple(map(int, joints[side]['elbow'][:2]))
        pt4 = tuple(map(int, joints[side]['wrist'][:2]))
        cv2.line(image, pt3, pt4, colors['elbow'], 2)

        # Hand connections
        if 'hand_wrist' in joints[side] and 'index_tip' in joints[side]:
            pt5 = tuple(map(int, joints[side]['hand_wrist'][:2]))
            pt6 = tuple(map(int, joints[side]['index_tip'][:2]))
            cv2.line(image, pt5, pt6, colors['wrist'], 2)

# Streamlit UI
st.title("Joint Angle Measurement")
st.markdown("""
**Analyzed Joints:**
- Neck flexion
- Shoulder elevation/flexion
- Elbow flexion
- Wrist dorsiflexion/radial deviation
- Trunk flexion
""")

uploaded_file = st.file_uploader("Upload Work Scene Image", type=["jpg", "png"])
threshold = st.slider("Set Risk Threshold (°)", 30, 90, 60)

if uploaded_file and uploaded_file.type.startswith("image"):
    img = Image.open(uploaded_file)
    img_np = np.array(img)

    # Handle RGBA images
    if img_np.shape[-1] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    processed_img, metrics = process_image(img_np)

    col1, col2 = st.columns(2)
    with col1:
        st.image(processed_img, channels="BGR", use_container_width=True)
    with col2:
        st.subheader("Joint Angle Analysis")
        for joint, angle in metrics['angles'].items():
            status = "⚠️" if angle > threshold else "✅"
            st.markdown(f"{status} ​**{joint}**: `{angle:.1f}°`")
else:
    st.info("Please upload a JPG/PNG image")
