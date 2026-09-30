from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg


LEFT_COLOR = (255, 120, 40)
RIGHT_COLOR = (40, 150, 255)
BODY_COLOR = (80, 220, 120)
FACE_COLOR = (220, 120, 220)

BODY_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_hand_root"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_hand_root"),
]

FACE_EDGES = [
    ("left_eye", "nose"),
    ("nose", "right_eye"),
    ("nose", "face-8"),
]


def hand_edges(side: str) -> list[tuple[str, str]]:
    root = f"{side}_hand_root"
    edges = []
    for finger in ("thumb", "forefinger", "middle_finger", "ring_finger", "pinky_finger"):
        joints = [root] + [f"{side}_{finger}{index}" for index in range(1, 5)]
        edges.extend(zip(joints, joints[1:]))
    return edges


LEFT_HAND_EDGES = hand_edges("left")
RIGHT_HAND_EDGES = hand_edges("right")


def normalized_frame_for_video_frame(
    frames: list[dict],
    video_frame_index: int,
    video_frame_count: int,
) -> dict | None:
    if not frames:
        return None
    if video_frame_count <= 1:
        return frames[0]
    index = round(video_frame_index * (len(frames) - 1) / (video_frame_count - 1))
    return frames[max(0, min(index, len(frames) - 1))]


def project_point(
    frame: dict,
    name: str,
    width: int,
    height: int,
    confidence_threshold: float,
) -> tuple[int, int, float] | None:
    item = (frame.get("keypoints") or {}).get(name) or {}
    xy = item.get("xy")
    score = float(item.get("score") or 0.0)
    if not xy or len(xy) != 2 or score < confidence_threshold:
        return None

    origin = frame.get("origin_xy")
    scale = frame.get("scale")
    if origin and len(origin) == 2 and scale and float(scale) > 0:
        x = float(xy[0]) * float(scale) + float(origin[0])
        y = float(xy[1]) * float(scale) + float(origin[1])
    else:
        # Tensor caches do not retain pixel-space normalization metadata.
        projection_scale = min(width, height) * 0.22
        x = width * 0.5 + float(xy[0]) * projection_scale
        y = height * 0.35 + float(xy[1]) * projection_scale
    return int(round(x)), int(round(y)), score


def draw_edges(
    image,
    frame: dict,
    edges: list[tuple[str, str]],
    color: tuple[int, int, int],
    confidence_threshold: float,
) -> None:
    height, width = image.shape[:2]
    for source, target in edges:
        first = project_point(frame, source, width, height, confidence_threshold)
        second = project_point(frame, target, width, height, confidence_threshold)
        if first is None or second is None:
            continue
        alpha = max(0.25, min(first[2], second[2]))
        edge_color = tuple(int(channel * alpha) for channel in color)
        cv2.line(image, first[:2], second[:2], edge_color, 2, cv2.LINE_AA)


def draw_points(
    image,
    frame: dict,
    names: list[str],
    color: tuple[int, int, int],
    confidence_threshold: float,
) -> None:
    height, width = image.shape[:2]
    for name in names:
        point = project_point(frame, name, width, height, confidence_threshold)
        if point is None:
            continue
        radius = 4 if "hand" not in name and "finger" not in name and "thumb" not in name else 3
        cv2.circle(image, point[:2], radius, color, -1, cv2.LINE_AA)


def render_keypoint_video(
    input_video: Path,
    normalized: dict,
    output_video: Path,
    confidence_threshold: float = 0.3,
) -> Path:
    intermediate_video = output_video.with_name(f"{output_video.stem}_mp4v.mp4")
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_video}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video dimensions: {input_video}")
    if fps <= 0:
        fps = 25.0

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(intermediate_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create overlay video: {intermediate_video}")

    frames = normalized.get("frames") or []
    body_names = sorted({name for edge in BODY_EDGES for name in edge})
    face_names = sorted({name for edge in FACE_EDGES for name in edge})
    left_names = sorted({name for edge in LEFT_HAND_EDGES for name in edge})
    right_names = sorted({name for edge in RIGHT_HAND_EDGES for name in edge})

    index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            normalized_frame = normalized_frame_for_video_frame(
                frames,
                index,
                frame_count,
            )
            if normalized_frame is not None and normalized_frame.get("valid", True):
                draw_edges(image, normalized_frame, BODY_EDGES, BODY_COLOR, confidence_threshold)
                draw_edges(image, normalized_frame, FACE_EDGES, FACE_COLOR, confidence_threshold)
                draw_edges(image, normalized_frame, LEFT_HAND_EDGES, LEFT_COLOR, confidence_threshold)
                draw_edges(image, normalized_frame, RIGHT_HAND_EDGES, RIGHT_COLOR, confidence_threshold)
                draw_points(image, normalized_frame, body_names, BODY_COLOR, confidence_threshold)
                draw_points(image, normalized_frame, face_names, FACE_COLOR, confidence_threshold)
                draw_points(image, normalized_frame, left_names, LEFT_COLOR, confidence_threshold)
                draw_points(image, normalized_frame, right_names, RIGHT_COLOR, confidence_threshold)

            cv2.putText(
                image,
                f"Frame {index + 1}/{max(frame_count, index + 1)}",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(image)
            index += 1
    finally:
        capture.release()
        writer.release()
    transcode_browser_video(intermediate_video, output_video)
    intermediate_video.unlink(missing_ok=True)
    return output_video


def transcode_browser_video(input_video: Path, output_video: Path) -> Path:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    output_video.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_video),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "FFmpeg could not create a browser-compatible video: "
            f"{completed.stderr[-1000:]}"
        )
    return output_video
