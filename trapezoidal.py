import argparse
import math
from logging import getLogger
from pathlib import Path

import cv2
import numpy as np

from utils import put_text_on_image, set_base_log_level


def _retrieve_args():
    parser = argparse.ArgumentParser(
        description="simple script to collect trapezoidallity of an image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="input image file",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        help="output image file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="increase output verbosity",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        help="decrease output verbosity",
    )

    args = parser.parse_args()

    args.verbose -= args.quiet
    del args.quiet

    return args


def draw_cross(image: np.ndarray, x: int, y: int) -> np.ndarray:
    img = image.copy()
    h, w, _ = img.shape
    cv2.line(img, (x, 0), (x, h - 1), (0, 255, 0))
    cv2.line(img, (0, y), (w - 1, y), (0, 255, 0))
    return img


def draw_lines(image: np.ndarray, plist: list[tuple[int, int]]) -> np.ndarray:
    img = image.copy()
    for i in range(len(plist)):
        cv2.circle(img, plist[i], 3, (0, 0, 255), 1)
        if 0 < i:
            cv2.line(img, plist[i], plist[i - 1], (0, 255, 0), 1)
    return img


def on_mouse(event, x, y, flags, params) -> None:
    img: np.ndarray = params["image"].copy()
    # wname: str = params["window_name"]
    plist: list[tuple[int, int]] = params["point_list"]

    # add point
    if event == cv2.EVENT_LBUTTONDOWN:
        plist.append((x, y))

    # delete point
    if event == cv2.EVENT_RBUTTONDOWN:
        plist.pop(-1)

    # show cross
    if event == cv2.EVENT_MOUSEMOVE:
        img = draw_cross(img, x, y)

    # draw points and lines
    img = draw_lines(img, plist)

    if 0 < len(plist):
        cv2.line(img, (x, y), plist[len(plist) - 1], (0, 255, 0), 1)

    # output text
    img = put_text_on_image(
        img, f"({x}, {y})", position=(x, y), font_scale=0.4
    )


def obtain_polygon(
    img: np.ndarray, window_name="click_image"
) -> list[tuple[int, int]]:
    params = {
        "image": img,
        "window_name": window_name,
        "point_list": [],
    }
    cv2.imshow(window_name, img)
    cv2.setMouseCallback(window_name, on_mouse, params)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    return params["point_list"]

    return img


def main(input_file: Path, output_file: Path, verbose: int) -> None:
    logger = getLogger(__name__)
    set_base_log_level(verbose)

    # 比率調整
    w_ratio = 1.1

    # 変換前4点の座標　p1:左上　p2:右上 p3:左下 p4:左下
    p1 = np.array([267, 960])
    p2 = np.array([2544, 378])
    p3 = np.array([216, 1494])
    # p4 = np.array([2592, 1053])

    # 入力画像の読み込み
    img = cv2.imread(str(input_file))

    # 　幅取得
    o_width = np.linalg.norm(p2 - p1)
    o_width = math.floor(o_width * w_ratio)

    # 　高さ取得
    o_height = np.linalg.norm(p3 - p1)
    o_height = math.floor(o_height)

    src_points = obtain_polygon(img)
    logger.debug("{src_points=}")

    # 変換後の4点
    dst = np.float32(
        [[0, 0], [o_width, 0], [0, o_height], [o_width, o_height]]
    )

    # 変換行列
    M = cv2.getPerspectiveTransform(src_points, dst)

    # 射影変換・透視変換する
    output = cv2.warpPerspective(img, M, (o_width, o_height))

    if output_file is not None:
        cv2.imwrite(str(output_file), output)


if __name__ == "__main__":
    main(**vars(_retrieve_args()))
