import argparse
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
        "--width_preview",
        type=int,
        default=500,
        help="set preview width",
    )
    parser.add_argument(
        "--border_width",
        type=int,
        default=100,
        help="preview border",
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
    wname: str = params["window_name"]
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
    cv2.imshow(wname, img)


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


def resize_img(img: np.ndarray, new_width: int) -> np.ndarray:
    height, width, _ = img.shape
    return cv2.resize(img, (new_width, int(height * new_width / width)))


def resize_and_bordering(
    img: np.ndarray,
    width_preview: int,
    border_width,
    color: tuple[int, int, int] = (250, 250, 250),
) -> np.ndarray:
    height, width, _ = img.shape
    ratio = width_preview / width
    border_height = border_width * height / width

    img_preview = resize_img(img, width_preview)

    img_preview_with_bordar = cv2.copyMakeBorder(
        img_preview,
        int(border_height),
        int(border_height),
        border_width,
        border_width,
        cv2.BORDER_CONSTANT,
        color,
    )

    def back_to_original_coordinate(
        data: tuple[float, float]
    ) -> tuple[int, int]:
        return (
            int((data[0] - border_width) / ratio),
            int((data[1] - border_height) / ratio),
        )

    return img_preview_with_bordar, back_to_original_coordinate


def obtain_similar_points(ps: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    change to average points
    """
    if abs(ps[0][0] - ps[1][0]) > abs(ps[0][0] - ps[3][0]):
        x_lt = (ps[0][0] + ps[3][0]) // 2
        x_rt = (ps[1][0] + ps[2][0]) // 2
        y_up = (ps[0][1] + ps[1][1]) // 2
        y_dn = (ps[2][1] + ps[3][1]) // 2
        return [(x_lt, y_up), (x_rt, y_up), (x_rt, y_dn), (x_lt, y_dn)]

    x_lt = (ps[0][0] + ps[1][0]) // 2
    x_rt = (ps[2][0] + ps[3][0]) // 2
    y_up = (ps[0][1] + ps[3][1]) // 2
    y_dn = (ps[1][1] + ps[2][1]) // 2
    return [
        (x_lt, y_up),
        (x_lt, y_dn),
        (x_rt, y_dn),
        (x_rt, y_up),
    ]


def get_rectangle(data: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [d[0] for d in data]
    ys = [d[1] for d in data]
    return min(xs), max(xs), min(ys), max(ys)


def warp_perspective_pack(
    img: np.ndarray,
    src_points: list[tuple[int, int]],
    dst_points: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    height, width, _ = img.shape
    M = cv2.getPerspectiveTransform(
        np.float32(src_points), np.float32(dst_points)
    )

    img_trapezoid = cv2.warpPerspective(img, M, (width, height))
    return img_trapezoid, M


def interactive_trapezoid(
    img: np.ndarray, width_preview: int, border_width: int
) -> np.ndarray:
    logger = getLogger(__name__).getChild("interactive_trapezoid")
    height, width, _ = img.shape

    # resize and bordaring
    img_preview, back_to_origi_coord = resize_and_bordering(
        img, width_preview, border_width
    )

    # obtain polygon on img_preview
    src_points_in_preview = obtain_polygon(img_preview)
    logger.debug(f"{src_points_in_preview=}")

    # obtain dst point using average coordinate
    dst_points_in_preview = obtain_similar_points(src_points_in_preview)

    # obtain points in original coordinate
    src_points = [back_to_origi_coord(pt) for pt in src_points_in_preview]
    dst_points = [back_to_origi_coord(pt) for pt in dst_points_in_preview]

    img_trapezoid, _ = warp_perspective_pack(img, src_points, dst_points)
    return img_trapezoid


def interactive_cropping(img: np.ndarray, width_preview: int) -> np.ndarray:
    # get preview image for cropping
    img_preview, back_to_origi_coord = resize_and_bordering(
        img, width_preview, 0
    )

    src_points_in_preview = obtain_polygon(img_preview)
    src_points = [back_to_origi_coord(pt) for pt in src_points_in_preview]
    x_min, x_max, y_min, y_max = get_rectangle(src_points)
    return img[y_min:y_max, x_min:x_max]


def scaled_imshow(
    img: np.ndarray, width_preview, wname: str = "scaled img"
) -> None:
    img_resized, _ = resize_and_bordering(img, width_preview, 0)
    cv2.imshow(wname, img_resized)
    cv2.waitKey(0)
    cv2.destroyWindow(wname)


def main(
    input_file: Path,
    output_file: Path,
    width_preview: int,
    border_width: int,
    verbose: int,
) -> None:
    logger = getLogger(__name__)
    set_base_log_level(verbose)

    img = cv2.imread(str(input_file))

    img_trapezoid = interactive_trapezoid(img, width_preview, border_width)

    img_trapezoid_cropped = interactive_cropping(img_trapezoid, width_preview)

    scaled_imshow(img_trapezoid_cropped, width_preview)

    if output_file is not None:
        cv2.imwrite(str(output_file), img_trapezoid_cropped)
        logger.info("output: {str(output_file)}")


if __name__ == "__main__":
    main(**vars(_retrieve_args()))
