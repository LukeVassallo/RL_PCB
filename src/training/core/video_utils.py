import cv2
from datetime import datetime
import numpy as np
import torch

def create_video(all_comp_grids,
                 ratsnest,
                 fileName=None,
                 v_id=None,
                 all_metrics=None,
                 draw_debug=False,
                 fps=30):
    width = all_comp_grids[0][0].shape[0]
    height = all_comp_grids[0][0].shape[1]
    channel = 1

    if all_metrics is not None:
        metrics_width = int(1*width)
        width = width + metrics_width

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    ts = datetime.now().strftime("%s_%f")

    if fileName is None:
        fileName = f"{ts}_video.mp4"

    video = cv2.VideoWriter(fileName, fourcc, float(
        fps), (width, height), False)

    if v_id is not None:
        for _ in range(fps):
            img = np.zeros(((width),height,1), np.uint8)
            (text_width, text_height) = cv2.getTextSize(text=f"{v_id}",
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 5,
                thickness=2
                )[0]

            cv2.putText(img,
            f"{v_id}",
            (int(0.5*width - text_width/2), int(0.5*height + text_height/2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            6,
            (128, 128, 0),
            3
            )
            video.write(img)

    for frame in range(len(all_comp_grids)):

        if all_metrics is not None:
            metrics_img = np.zeros((height, metrics_width, channel),
                                   dtype = np.uint8)

        img = all_comp_grids[frame][0] + \
            2*all_comp_grids[frame][1]

        if draw_debug is True:
            img = np.maximum(img,all_comp_grids[frame][2])

        if len(ratsnest) != 0:
            img = np.maximum(img, ratsnest[frame])

        cv2.putText(img, f"{frame}",
                    (int(0.075*width), int(0.1*height)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, # 0.85 is the font scale
                    (128, 128, 0),
                    2)

        accumulated_reward = 0
        if all_metrics is not None and frame > 0:
            height_mult = 0.04
            total_cost = 0
            total_reward = 0
            total_nodes = 0
            for item in all_metrics[frame-1]:
                # For five components
                cv2.putText(metrics_img,
                            f"id; cost    : {item['id']} ({item['name']}); {np.round(item['weighted_cost'],2)} ({np.round(item['reward'],2)})",
                            (int(0.02*width), int(height_mult*height)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (128, 128, 0),
                            1)
                height_mult += 0.04
                cv2.putText(metrics_img,
                            f"rW; rHPWL   : {np.round(item['W'],2)} ({np.round(item['We'],2)}); {np.round(item['HPWL'],2)} ({np.round(item['HPWLe'],2)})", (int(0.02*width), int(height_mult*height)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (128, 128, 0),
                            1)
                height_mult += 0.04
                cv2.putText(metrics_img,
                            f"ol           : {np.round(item['ol'],2)}", (int(0.02*width), int(height_mult*height)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (128, 128, 0),
                            1)

                total_cost += item["weighted_cost"]
                total_reward += item["reward"]
                total_nodes += 1
                height_mult += 0.075

            cv2.putText(metrics_img,
                        f"Average cost        : {np.round(total_cost/total_nodes,2)}",
                        (int(0.02*width), int(0.85*height)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (128, 128, 0),
                        1)
            cv2.putText(metrics_img,
                        f"Average reward      : {np.round(total_reward/total_nodes,2)}",
                        (int(0.02*width), int(0.9*height)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (128, 128, 0),
                        1)
            accumulated_reward += total_reward/total_nodes
            cv2.putText(metrics_img,
                        f"Accumulated reward      : {np.round(accumulated_reward,2)}",
                        (int(0.02*width), int(0.95*height)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (128, 128, 0),
                        1)

        if all_metrics is not None:
            metrics_img = np.reshape(metrics_img,
                                    (metrics_img.shape[0],metrics_img.shape[1])
                                    )

        if all_metrics is not None:
            video.write(cv2.hconcat([img,metrics_img]))
        else:
            video.write(img)

    video.release()

def video_frames(all_comp_grids, ratsnest, v_id=None):
    width = all_comp_grids[0][0].shape[0]
    height = all_comp_grids[0][0].shape[1]
    channels = 1

    fps = 30
    v_id_duration_in_frames = int(fps/2)

    total_frames = len(all_comp_grids) + v_id_duration_in_frames

    frame_buffer = np.zeros((total_frames, height, width, channels), np.uint8)

    if v_id is not None:
        for i in range(v_id_duration_in_frames):
            (text_width, text_height) = cv2.getTextSize(text=f"{v_id}",
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 5,
                thickness=2
                )[0]

            cv2.putText(frame_buffer[i],
            f"{v_id}",
            (int(0.5*width - text_width/2), int(0.5*height + text_height/2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            6,
            (128, 128, 0),
            3
            )

    for frame in range(len(all_comp_grids)):
        idx = frame+v_id_duration_in_frames
        frame_buffer[idx] = np.resize(
            np.maximum(all_comp_grids[frame][0] + 2*all_comp_grids[frame][1],
                       ratsnest[frame]),
            (width,height,channels)
            )

        cv2.putText(frame_buffer[idx], f"{frame}",
                    (int(0.075*width), int(0.1*height)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (128, 128, 0),
                    2)

    return frame_buffer

def write_frame_buffer(frame_buffer, fileName=None):

    width = frame_buffer[0].shape[-2]
    height = frame_buffer[0].shape[-3]
    fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    ts = datetime.now().strftime("%s_%f")

    if fileName is None:
        fileName = f"{ts}_video.mp4"

    video = cv2.VideoWriter(fileName,
                            fourcc,
                            float(fps),
                            (width, height),
                            False)

    for frame in frame_buffer:
        video.write(frame)

    video.release()

def create_image(all_comp_grids, ratsnest, fileName=None, draw_debug=False):

    img = all_comp_grids[-1][0] + 2*all_comp_grids[-1][1]

    if draw_debug is True:
        img = np.maximum(img,all_comp_grids[-1][2])

    if len(ratsnest) != 0:
        img = np.maximum(img, ratsnest[-1])

    cv2.imwrite(fileName, img)

def get_video_tensor(all_comp_grids, ratsnest):
    width = all_comp_grids[0][0].shape[0]
    height = all_comp_grids[0][0].shape[1]
    channels = 3
    frame_buf = []
    frames = len(all_comp_grids)

    for frame_number in range(frames):
        frame = all_comp_grids[frame_number][0] + \
             2*all_comp_grids[frame_number][1]
        frame = np.maximum(frame, ratsnest[frame_number])

        cv2.putText(frame, f"{frame_number}",
                    (int(0.075*width),int(0.1*height)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,    # 0.85 is the font scale
                    (128, 128, 0),
                    2)

        frame = np.tile(frame, (channels,1,1))
        np.reshape(frame, (channels, height, width))
        frame_buf.append(frame)

    video_tensor = torch.tensor(np.array(frame_buf))
    video_tensor = video_tensor.view([1,frames,channels,height,width])
    return video_tensor
