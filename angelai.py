import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

# 初始化模型
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def get_coord(landmark, model_type='pose', img_width=640, img_height=480):
    """统一三维坐标处理（手部z轴补零）"""
    if model_type == 'pose':
        return [landmark.x * img_width, landmark.y * img_height, landmark.z * img_width]
    elif model_type == 'hands':
        return [landmark.x * img_width, landmark.y * img_height, 0]  # 手部深度补零

def calculate_angle(a, b, c, plane='sagittal'):
    """安全的三维角度计算"""
    try:
        # 强制三维化
        a = np.array(a)[:3].astype('float64')
        b = np.array(b)[:3].astype('float64')
        c = np.array(c)[:3].astype('float64')

        # 向量计算
        ba = a - b
        bc = c - b

        # 平面投影
        if plane == 'sagittal':
            ba = np.array([0, ba[1], ba[2]])
            bc = np.array([0, bc[1], bc[2]])
        elif plane == 'frontal':
            ba = np.array([ba[0], 0, ba[2]])
            bc = np.array([bc[0], 0, bc[2]])
        elif plane == 'transverse':
            ba = ba[:2]
            bc = bc[:2]

        # 零向量处理
        ba_norm = np.linalg.norm(ba)
        bc_norm = np.linalg.norm(bc)
        if ba_norm < 1e-6 or bc_norm < 1e-6:
            return 0.0

        # 角度计算
        cosine = np.dot(ba, bc) / (ba_norm * bc_norm)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    except Exception as e:
        print(f"角度计算错误: {str(e)}")
        return 0.0

def calculate_neck_flexion(nose, shoulder_mid, hip_mid):
    """计算颈部前屈角度（偏离中心位的角度）"""
    try:
        # 将坐标转换为 numpy 数组
        nose = np.array(nose)[:2]  # 只取 x 和 y 坐标
        shoulder_mid = np.array(shoulder_mid)[:2]
        hip_mid = np.array(hip_mid)[:2]

        # 计算躯干轴线（肩膀中点到髋部中点）
        torso_vector = hip_mid - shoulder_mid
        torso_angle = np.degrees(np.arctan2(torso_vector[1], torso_vector[0]))

        # 计算头部向量（鼻子到肩膀中点）
        head_vector = nose - shoulder_mid
        head_angle = np.degrees(np.arctan2(head_vector[1], head_vector[0]))

        # 计算偏离中心位的角度
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
        print(f"颈部前屈计算错误: {str(e)}")
        return 0.0

def calculate_trunk_flexion(shoulder_mid, hip_mid, knee_mid):
    """计算背部屈曲角度（偏离中心位的角度）"""
    try:
        # 计算躯干向量（肩膀中点到髋部中点）
        torso_vector = np.array(hip_mid) - np.array(shoulder_mid)
        torso_angle = np.degrees(np.arctan2(torso_vector[1], torso_vector[0]))

        # 计算腿部向量（髋部中点到膝部中点）
        leg_vector = np.array(knee_mid) - np.array(hip_mid)
        leg_angle = np.degrees(np.arctan2(leg_vector[1], leg_vector[0]))

        # 计算背部屈曲角度：躯干向量与腿部向量的夹角
        flexion_angle = leg_angle - torso_angle

        # 规范化角度到 0-180 度范围
        if flexion_angle < 0:
            flexion_angle += 360
        if flexion_angle > 180:
            flexion_angle = 360 - flexion_angle

        return flexion_angle
    except Exception as e:
        print(f"背部屈曲计算错误: {str(e)}")
        return 0.0

def process_image(image):
    H, W, _ = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 并行处理模型
    pose_result = pose.process(img_rgb)
    hands_result = hands.process(img_rgb)

    metrics = {'angles': {}}

    if pose_result.pose_landmarks:
        # 关键点获取
        def get_pose_pt(landmark):
            return get_coord(pose_result.pose_landmarks.landmark[landmark], 'pose', W, H)

        # 基础关节点
        joints = {
            'left side': {
                'Shoulder': get_pose_pt(mp_pose.PoseLandmark.LEFT_SHOULDER),
                'Elbow': get_pose_pt(mp_pose.PoseLandmark.LEFT_ELBOW), {
                'Wrists': get_pose_pt(mp_pose.PoseLandmark.LEFT_WRIST),
                'Hip': get_pose_pt(mp_pose.PoseLandmark.LEFT_HIP),
                'knee': get_pose_pt(mp_pose.PoseLandmark.LEFT_KNEE)
            },
            'right side': {
                'Shoulder': get_pose_pt(mp_pose.PoseLandmark.RIGHT_SHOULDER),
                'Elbows': get_pose_pt(mp_pose.PoseLandmark.RIGHT_ELBOW),
                'Wrists': get_pose_pt(mp_pose.PoseLandmark.RIGHT_WRIST),
                'Hip': get_pose_pt(mp_pose.PoseLandmark.RIGHT_HIP),
                'Knees': get_pose_pt(mp_pose.PoseLandmark.RIGHT_KNEE)
            },
            'mid': {
                'shoulders': [(get_pose_pt(mp_pose.PoseLandmark.LEFT_SHOULDER)[i] +
                          get_pose_pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)[i]) / 2 for i in range(3)],
                'Hip': [(get_pose_pt(mp_pose.PoseLandmark.LEFT_HIP)[i] +
                          get_pose_pt(mp_pose.PoseLandmark.RIGHT_HIP)[i]) / 2 for i in range(3)],
                'knee': [(get_pose_pt(mp_pose.PoseLandmark.LEFT_KNEE)[i] +
                         get_pose_pt(mp_pose.PoseLandmark.RIGHT_KNEE)[i]) / 2 for i in range(3)]
            },
            'Nose': get_pose_pt(mp_pose.PoseLandmark.NOSE)
        }
        # 合并手部数据
        if hands_result.multi_hand_landmarks:
            for hand in hands_result.multi_hand_landmarks:
                side = 'left' if hand.landmark[0].x < 0.5 else 'right'
                joints[side].update({
                    'Wrist': get_coord(hand.landmark[mp_hands.HandLandmark.WRIST], 'hands', W, H),
                    'Index Finger Middle Joint': get_coord(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP], 'hands', W, H),
                    'Index Finger Tip': get_coord(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP], 'hands', W, H)
                })

        # 计算指定关节角度
        try:
            metrics['angles']['Neck forward flexion'] = calculate_neck_flexion(
                joints['nose'], joints['mid']['shoulder'], joints['mid']['hip'])
         
            # 肩部运动
            for side in ['left', 'right']:
                # Supination (coronal plane)
                metrics['angles'][f'{side.capitalize()} shoulder supination'] = calculate_angle(
                    joints[side]['hips'], joints[side]['shoulders'], joints[side]['elbows'], 'frontal')
                # Anterior extension (sagittal plane)
                metrics['angles'][f'{side.capitalize()} shoulder forward'] = calculate_angle(
                    joints[side]['hip'], joints[side]['shoulder'], joints[side]['elbow'], 'sagittal')
        
            # Elbow flexion
            for side in ['left', 'right']:
                metrics['angles'][f'{side.capitalize()} elbow flexion'] = calculate_angle(
                    joints[side]['shoulder'], joints[side]['elbow'], joints[side]['wrist'], 'sagittal')
        
            # Wrist movements
            for side in ['left', 'right']:
                if 'wrist' in joints[side]:
                    # dorsal extension
                    metrics['angles'][f'{side.capitalize()} wrist dorsal extension'] = calculate_angle(
                        joints[side]['elbow'], joints[side]['wrist'].
                        joints[side].get('tip of index finger', [0, 0, 0]), 'sagittal')
                    # radial deviation
                    metrics['angles'][f'{side.capitalize()} wrist radial deviation'] = calculate_angle(
                        joints[side]['index finger middle joint'], joints[side]['wrist'].
                        joints[side].get('tip of index finger', [0, 0, 0]), 'frontal')
        
        except Exception as e:
            print(f"错误发生: {e}")
        
            # 背部屈曲
            metrics['angles']['背部屈曲'] = calculate_trunk_flexion(
                joints['mid']['肩膀'], joints['mid']['臀部'], joints['mid']['膝部'])

            # 可视化
            draw_landmarks(image, joints)

        except KeyError as e:
            print(f"关键点缺失: {str(e)}")

    return image, metrics


def draw_landmarks(image, joints):
    """可视化指定关节连线"""
    # 颜色配置
    colors = {
        'neck': (255, 200, 0),  # 金黄色
        'shoulder': (0, 255, 0),  # 绿色
        'elbow': (0, 255, 255),  # 青色
        'wrist': (255, 0, 255)  # 品红色
    }

    # 绘制颈部前屈
    nose = tuple(map(int, joints['鼻子'][:2]))
    shoulder_mid = tuple(map(int, joints['mid']['肩膀'][:2]))
    hip_mid = tuple(map(int, joints['mid']['臀部'][:2]))
    cv2.line(image, nose, shoulder_mid, colors['neck'], 2)
    cv2.line(image, shoulder_mid, hip_mid, colors['neck'], 2)

    # 绘制上肢
    for side in ['左侧', '右侧']:
        # 肩-肘
        pt1 = tuple(map(int, joints[side]['肩膀'][:2]))
        pt2 = tuple(map(int, joints[side]['肘部'][:2]))
        cv2.line(image, pt1, pt2, colors['shoulder'], 2)

        # 肘-腕
        pt3 = tuple(map(int, joints[side]['肘部'][:2]))
        pt4 = tuple(map(int, joints[side]['手腕'][:2]))
        cv2.line(image, pt3, pt4, colors['elbow'], 2)

        # 手部连线（如果存在食指尖端，绘制手部线条）
        if '手腕' in joints[side] and '食指尖端' in joints[side]:
            pt5 = tuple(map(int, joints[side]['手腕'][:2]))
            pt6 = tuple(map(int, joints[side]['食指尖端'][:2]))  # 修正bug
            cv2.line(image, pt5, pt6, colors['wrist'], 2)

            
# Streamlit interface
st.title("Angle Analysis System")
st.markdown("""
**Analyze joints:**
- Neck forward flexion
- Shoulder supination/extension
- Elbow flexion/extension
- Dorsal extension/radial deviation of the wrist
- Dorsiflexion
""")

uploaded_file = st.file_uploader("Upload work scene image", type=["jpg", "png"])
threshold = st.slider("Set risk threshold (°)", 30, 90, 60)
if uploaded_file and uploaded_file.type.startswith("image"):
    img = Image.open(uploaded_file)
    img_np = np.array(img)

    # Process RGBA images
    if img_np.shape[-1] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else: 
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    processed_img, metrics = process_image(img_np)

    # Two-column layout
    col1, col2 = st.columns(2)
    with col1:
        st.image(processed_img, channels="BGR", use_container_width=True)

    with col2:
        st.subheader("Joint angle analysis")
        for joint, angle in metrics['angles'].items():
            status = "⚠️"if angle > threshold else "✅"
            st.markdown(f"{status} **{joint}**:`{angle:.1f}°`")
else:
    st.info("Please upload an image in JPG/PNG format")
