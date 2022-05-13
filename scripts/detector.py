import sys

sys.path.append('.')
from od.models.modules.experimental import *
from od.data.datasets import letterbox
from utils.general import *
from utils.split_detector import SPLITINFERENCE
from utils.torch_utils import *

import argparse
import os
from pathlib import Path
import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'

def plot_image(ids):
  cl = int(ids)
  if cl == 0: return (0,255,0)
  elif cl == 1: return (0,255,255)
  elif cl == 2: return (255,0,255)
  else: return (255,255,0)


@torch.no_grad()
def run(
        weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (height, width)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        device='cuda',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_txt=False,  # save results to *.txt
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,
):
    device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
    model = attempt_load(weights, map_location=device)  # load FP32 model
    
    def detect_image(image):
        bboxes = []
        bboxes_c = []
        scores = []
        ids = []
        im0s = image
        W, H = image.shape[:2]
        # print(image.shape)
        img = letterbox(im0s, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]
        # print('pred: ',pred.size())
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes,
                                    agnostic=agnostic_nms)
        
        for i, det in enumerate(pred):
            if det is not None and len(det):
                # print(det)
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in det:
                    x_min = xyxy[0].cpu()
                    y_min = xyxy[1].cpu()
                    x_max = xyxy[2].cpu()
                    y_max = xyxy[3].cpu()
                    score = conf.cpu()
                    clas = cls.cpu()
                    # print(clas)
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    bboxes.append([x_min, y_min, x_max, y_max])
                    scores.append(score)
                    ids.append(clas)
                    bboxes_c.append([(x_min + w / 2)/W, (y_min + h / 2)/H, w/W, h/H])
        if save_txt:  return np.asarray(bboxes_c), np.asarray(bboxes), np.asarray(scores), np.asarray(ids)
        else: return np.asarray(bboxes), np.asarray(scores), np.asarray(ids)


    def plot_txt(img, path):
      if save_txt:
        bboxes_c, bboxes, scores, ids = detect_image(im)
        for idx in range(bboxes.shape[0]):
            bbox_c = bboxes_c[idx]
            bbox = bboxes[idx].astype(int)
            score = scores[idx]
            cl = ids[idx]
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), plot_image(cl), 2, 2)
            cv2.putText(im, str(int(cl)), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, plot_image(cl), 2, cv2.LINE_AA)
            with open(save_dir / 'labels' / (path.split('.')[0] + '.txt'), 'a') as f:
              f.write('{} {} {} {} {}\n'.format(int(cl), bbox_c[0],bbox_c[1],bbox_c[2],bbox_c[3]))
        print(path)
        return im

      else:
        bboxes, scores, ids = detect_image(im)
        for idx in range(bboxes.shape[0]):
            bbox = bboxes[idx].astype(int)
            score = scores[idx]
            cl = ids[idx]
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), plot_image(cl), 2, 2)
            cv2.putText(im, str(int(cl)), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, plot_image(cl), 2, cv2.LINE_AA)
        print(path)
        return im


    # Directories
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    i=0
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    

    source = str(source)

    is_file = Path(source).suffix[1:] in (IMG_FORMATS)
    

    if is_file:
      img = source.split('/')[-1]
      im = cv2.imread(source)
      plot_txt(im, img)
      cv2.imwrite(os.path.join(save_dir, img), im)
      

    else:
      imgs = os.listdir(source)
      for img in imgs:
        im = cv2.imread(os.path.join(source, img))
        plot_txt(im, img)
        cv2.imwrite(os.path.join(save_dir, img), im)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
