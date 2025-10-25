"""
demo_video.py
- 파일 동영상(mp4/avi)을 프레임 단위로 읽어 UFLD-v2 추론을 수행하고,
  차선 포인트를 프레임에 시각화하여 다시 동영상으로 저장하는 스크립트.

핵심 포인트
1) 학습 전처리와 동일한 입력 크기를 맞추기 위해
   - 세로를 int(train_height / crop_ratio) 까지 Resize
   - 그 다음 하단 기준(bottom)으로 train_height만큼 크롭
   - Normalize (ImageNet 통계)
2) 디바이스-중립: .cuda() 대신 .to(device) 사용 (CPU-only 환경 호환)
3) state_dict 로드 시 'module.' prefix 제거(멀티GPU 학습 호환)
"""

import os, sys, cv2, torch, argparse
from PIL import Image

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from utils.common import merge_config, get_model
from utils.dist_utils import dist_print


def pred2coords(pred, row_anchor, col_anchor, local_width = 1, original_image_width = 1640, original_image_height = 590):
    
    """
    UFLD의 하이브리드(행/열) 앵커 기반 출력(pred)을
    원본 이미지의 (x, y) 픽셀 좌표로 복원.

    - 행(row) 축: y 위치는 row_anchor[k] * H, x 위치는 grid softmax 가중 평균으로 서브그리드 정밀화
    - 열(col) 축: x 위치는 col_anchor[k] * W, y 위치는 grid softmax 가중 평균으로 서브그리드 정밀화
    """
    
    # 출력 텐서 차원 파악
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape
    
    # 각각의 축에서 "그리드 argmax" 및 "존재성" 판정
    # argmax(1): grid 차원에서 최댓값 인덱스를 선택 → 거친 위치 후보
    max_indices_row = pred['loc_row'].argmax(1).cpu()
    valid_row = pred['exist_row'].argmax(1).cpu()    # 존재 여부 분류 결과 (행 방향)
    
    max_indices_col = pred['loc_col'].argmax(1).cpu()
    valid_col = pred['exist_col'].argmax(1).cpu()   # 존재 여부 분류 결과 (열 방향)
    
    # 이후 연산을 CPU 텐서로
    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()
    
    coords = []
    
    # 데모 관례: 가운데 2개 차선을 row 축, 양 끝 2개를 col 축으로 복원
    row_lane_idx = [1,2]
    col_lane_idx = [0,3]

    # --- 행(row) 축 기반 복원 ---
    for i in row_lane_idx:
        tmp = []
        # 절반 이상 앵커에서 "존재"로 판단될 때만 유효한 차선으로 간주
        if valid_row[0,:,i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):     # 각 row 앵커 k (세로 위치 라인)
                if valid_row[0,k,i]:
                    # argmax 근방(local_width) 범위를 softmax 가중 평균으로 서브셀 정밀화
                    all_ind = torch.arange(max(0,max_indices_row[0,k,i]-local_width),
                                           min(num_grid_row-1, max_indices_row[0,k,i]+local_width)+1)
                    
                    out_tmp = (pred['loc_row'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                    # 정규화(0~num_grid_row-1) → 원본 가로폭 스케일링
                    out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                    # x는 가로 좌표, y는 row_anchor에서 높이 스케일링 (행 앵커는 y 위치 의미)
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))    
        coords.append(tmp)

    # --- 열(col) 축 기반 복원 ---
    for i in col_lane_idx:
        tmp = []
        # (데모 기준) 1/4 이상 앵커에서 "존재"로 판단 시 유효
        if valid_col[0,:,i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):    # 각 col 앵커 k (가로 위치 라인)
                if valid_col[0,k,i]:
                    all_ind = torch.arange(max(0,max_indices_col[0,k,i]-local_width),
                                           min(num_grid_col-1, max_indices_col[0,k,i]+local_width)+1)
                    
                    out_tmp = (pred['loc_col'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                    # 정규화(0~num_grid_col-1) → 원본 세로높이 스케일링
                    out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                    # x는 col_anchor에서 가로 스케일링 (열 앵커는 x 위치 의미), y는 가중 평균 값
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
        coords.append(tmp)
    
    return coords



if __name__ == "__main__":
    # 기존 config / 가중치 인자 유지: configs/culane_res18.py --test_model ...
    video_argp = argparse.ArgumentParser(add_help=False)
    video_argp.add_argument('--video_path', type=str, required=True)    # 입력 비디오 경로
    video_argp.add_argument('--out_path', type=str, default='ufld_out.mp4')     # 출력 비디오 경로
    known_args, remaining = video_argp.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining    # 나머지 인자는 merge_config가 처리하게 재구성

    args, cfg = merge_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # CPU/GPU 자동 선택(코드 상 통일)

    net = get_model(cfg)    # cfg에 맞춰 네트워크 생성
    # 체크포인트는 CPU로 우선 로드(장치 독립); dict 안에 'model' 키가 있는 형식에 대응
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    # DataParallel 호환: 'module.' prefix 제거
    state_dict = { (k[7:] if k.startswith('module.') else k): v for k,v in state_dict.items() }
    net.load_state_dict(state_dict, strict=False)   # 키 일부 불일치 허용
    net.to(device).eval()   # 디바이스로 이동 후 평가 모드

    # 학습과 동일한 입력 크기를 위해: Resize( H/crop_ratio, W ) 후 "하단 기준"으로 H를 크롭
    resize_h = int(cfg.train_height / cfg.crop_ratio)
    def bottom_crop(pil_img):
        # crop()은 (top, left, height, width)를 받음. top은 '위에서부터의' 오프셋.
        # top = resize_h - train_height → 위쪽 strip이 잘리고, 아래쪽 기준 H만큼 남음.
        top = resize_h - cfg.train_height  # 하단 기준 크롭의 상단 오프셋
        return TF.crop(pil_img, top=top, left=0,
                       height=cfg.train_height, width=cfg.train_width)

    img_transforms = transforms.Compose([
        transforms.Resize((resize_h, cfg.train_width)),     # 1) 세로를 늘리고(학습 대비 crop_ratio 반영)
        transforms.Lambda(bottom_crop),                     # 2) 하단 기준으로 H만큼 크롭(학습 전처리와 일치)
        transforms.ToTensor(),                              # 3) [0,1] tensor로 변환 (PIL→Tensor)
        transforms.Normalize((0.485, 0.456, 0.406),         # 4) ImageNet mean/std로 표준화
                            (0.229, 0.224, 0.225)),
    ])

    cap = cv2.VideoCapture(known_args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {known_args.video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0         # FPS가 0/NaN이면 기본 30fps
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))     # 원본 프레임 폭
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    # 원본 프레임 높이
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # mp4 코덱(환경에 따라 다를 수 있음)
    vout = cv2.VideoWriter(known_args.out_path, fourcc, fps, (W, H))    # 출력 비디오 스트림 생성

    with torch.no_grad():   # 추론 시 그래디언트 비활성화
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break   # 영상 끝
            # OpenCV는 BGR → torchvision은 RGB 가정 → RGB로 변환 후 PIL로 래핑
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)   # BGR -> RGB
            pil = Image.fromarray(frame_rgb)
            # (Resize → bottom crop → Tensor/Normalize) 전처리 후 [1,C,H,W]로 배치 축 추가
            inp = img_transforms(pil).unsqueeze(0).to(device)
            pred = net(inp)  # 네트워크 추론

            # 원본 프레임 크기(W,H) 기준으로 좌표 복원(시각화는 원본 좌표계로)
            coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor,
                                 original_image_width=W, original_image_height=H)
            vis = frame_bgr.copy()
            for lane in coords:
                for (x, y) in lane:
                    # 차선 포인트를 초록색 점(반지름=5)으로 표시
                    cv2.circle(vis, (x, y), 5, (0,255,0), -1)
            vout.write(vis)   # 프레임 기록

    cap.release(); vout.release()
    dist_print(f"Video saved to: {known_args.out_path}")
