#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_webcam.py — UFLD-v2 webcam inference (CPU/GPU 겸용)

- 웹캠 프레임을 읽어 전처리(Resize -> bottom crop -> ToTensor/Normalize) 후
  네트워크 추론, pred2coords로 복원, 프레임에 시각화하여 표시/저장.
- 디바이스-중립(.to(device))으로 동작. (레포 내 .cuda() 강제 호출이 남아 있다면
  model/*.py의 .cuda()를 제거해야 CPU-only에서도 동작합니다.)
"""

import os, sys, time, cv2, torch, argparse
from PIL import Image

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from utils.common import merge_config, get_model
from utils.dist_utils import dist_print


# ---------------------------
# 1) UFLD 좌표 복원 유틸
# ---------------------------
def pred2coords(pred,
                row_anchor,
                col_anchor,
                local_width=1,
                original_image_width=1640,
                original_image_height=590):
    """
    UFLD 하이브리드(행/열) 앵커 기반 출력(pred)을 원본 (x,y) 픽셀 좌표로 복원.
    - 행(row)축: y는 row_anchor[k]*H, x는 grid softmax 가중평균
    - 열(col)축: x는 col_anchor[k]*W, y는 grid softmax 가중평균
    """
    bs, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    bs, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    # grid argmax & 존재성 판정
    max_indices_row = pred['loc_row'].argmax(1).cpu()   # [N, num_cls_row, num_lane_row]
    valid_row       = pred['exist_row'].argmax(1).cpu() # [N, num_cls_row, num_lane_row]
    max_indices_col = pred['loc_col'].argmax(1).cpu()
    valid_col       = pred['exist_col'].argmax(1).cpu()

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []
    # 데모 관례: 가운데 2개 차선을 row 축, 양 끝 2개를 col 축으로 복원
    row_lane_idx = [1, 2]
    col_lane_idx = [0, 3]

    # 행(row) 축 복원
    for i in row_lane_idx:
        tmp = []
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    center = int(max_indices_row[0, k, i])
                    lo = max(0, center - local_width)
                    hi = min(num_grid_row - 1, center + local_width)
                    all_ind = torch.arange(lo, hi + 1, dtype=torch.long)
                    out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row - 1) * original_image_width  # x
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))  # (x,y)
        coords.append(tmp)

    # 열(col) 축 복원
    for i in col_lane_idx:
        tmp = []
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    center = int(max_indices_col[0, k, i])
                    lo = max(0, center - local_width)
                    hi = min(num_grid_col - 1, center + local_width)
                    all_ind = torch.arange(lo, hi + 1, dtype=torch.long)
                    out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_col - 1) * original_image_height  # y
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))  # (x,y)
        coords.append(tmp)

    return coords


# ---------------------------
# 2) 전처리 파이프라인
# ---------------------------
def build_transforms(cfg):
    """
    학습과 동일한 입력 크기:
      - Resize: (int(train_height / crop_ratio), train_width)
      - Bottom Crop: 위에서 top=resize_h-train_height만큼 잘라 하단부만 남김
      - ToTensor & Normalize (ImageNet 통계)
    """
    resize_h = int(cfg.train_height / cfg.crop_ratio)

    def bottom_crop(pil_img):
        top = resize_h - cfg.train_height  # 상단 strip 제거 → 하단 기준 영역 유지
        return TF.crop(pil_img, top=top, left=0,
                       height=cfg.train_height, width=cfg.train_width)

    return transforms.Compose([
        transforms.Resize((resize_h, cfg.train_width)),
        transforms.Lambda(bottom_crop),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])


# ---------------------------
# 3) 비디오 저장 도우미
# ---------------------------
def open_writer(out_path, fps, width, height):
    """
    mp4v 우선 시도, 실패 시 MJPG(AVI) 폴백.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if writer.isOpened():
        return writer, out_path

    base, _ = os.path.splitext(out_path)
    alt_path = base + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(alt_path, fourcc, fps, (width, height))
    if writer.isOpened():
        dist_print(f"[warn] mp4v가 동작하지 않아 AVI(MJPG)로 저장합니다: {alt_path}")
        return writer, alt_path

    raise RuntimeError("VideoWriter 초기화 실패: mp4v, MJPG 모두 열 수 없습니다.")


# ---------------------------
# 4) 메인
# ---------------------------
def main():
    # a) 웹캠/출력 인자만 선파싱(나머지는 merge_config가 처리)
    argp = argparse.ArgumentParser(add_help=False)
    argp.add_argument('--cam_index', type=int, default=0, help='웹캠 인덱스 (기본 0)')
    argp.add_argument('--width', type=int, default=None, help='웹캠 요청 폭 (옵션)')
    argp.add_argument('--height', type=int, default=None, help='웹캠 요청 높이 (옵션)')
    argp.add_argument('--out_path', type=str, default=None, help='저장 파일 경로(예: out.mp4). 생략 시 저장 안 함')
    argp.add_argument('--show_fps', action='store_true', help='FPS 오버레이 표시')
    argp.add_argument('--window', type=str, default='UFLD Webcam', help='표시 윈도우 이름')
    known, remaining = argp.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    # b) UFLD 설정/장치
    args, cfg = merge_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dist_print(f"[info] device = {device}")

    # c) 모델 로드
    net = get_model(cfg)
    ckpt = torch.load(cfg.test_model, map_location='cpu')
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
    net.load_state_dict(state_dict, strict=False)
    net.to(device).eval()

    # d) 전처리
    tfm = build_transforms(cfg)

    # e) 웹캠 열기
    cap = cv2.VideoCapture(known.cam_index)  # 장치 인덱스(0,1,...) 또는 스트림 URL/파이프 가능
    if known.width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, known.width)
    if known.height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, known.height)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠을 열 수 없습니다: index={known.cam_index}")

    # 실제 프레임 크기/FPS 확인
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dist_print(f"[info] webcam opened: {W}x{H} @ ~{fps:.1f}fps")

    # f) 저장 설정(선택)
    writer = None
    out_path_final = None
    if known.out_path:
        writer, out_path_final = open_writer(known.out_path, fps, W, H)
        dist_print(f"[info] recording to: {out_path_final}")

    cv2.namedWindow(known.window, cv2.WINDOW_NORMAL)

    # g) 추론 루프
    t_last = time.perf_counter()
    frame_count, fps_ema = 0, 0.0
    alpha = 0.1  # EMA 계수

    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # BGR -> RGB (torchvision 전처리는 RGB 가정)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)

            # Resize -> bottom crop -> Tensor/Normalize
            inp = tfm(pil).unsqueeze(0).to(device)   # [1,3,Hcfg,Wcfg]

            # 추론
            pred = net(inp)

            # 좌표 복원(원본 프레임 크기 기준)
            coords = pred2coords(
                pred,
                cfg.row_anchor,
                cfg.col_anchor,
                original_image_width=W,
                original_image_height=H
            )

            # 시각화
            vis = frame_bgr.copy()
            for lane in coords:
                for (x, y) in lane:
                    cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)

            # FPS 표시(옵션)
            if known.show_fps:
                t_now = time.perf_counter()
                inst_fps = 1.0 / max(t_now - t_last, 1e-6)
                fps_ema = inst_fps if frame_count == 0 else (1 - alpha) * fps_ema + alpha * inst_fps
                t_last = t_now
                frame_count += 1
                cv2.putText(vis, f"{fps_ema:5.1f} FPS", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

            # 화면/저장
            if writer is not None:
                writer.write(vis)
            cv2.imshow(known.window, vis)

            # q 또는 ESC로 종료
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    # h) 종료 처리
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if out_path_final:
        dist_print(f"[done] saved to: {out_path_final}")


if __name__ == "__main__":
    # cuDNN benchmark는 GPU에서만 효과. 켜두어도 무해.
    torch.backends.cudnn.benchmark = True
    main()
