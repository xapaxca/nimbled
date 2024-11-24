import cv2
from multiprocessing import Pool, cpu_count
import os
import time
import logging
import yaml


def setup_logger():
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=[logging.FileHandler("frame_extraction.log"), logging.StreamHandler()])

def process_chunk(args):
    video_path, output_frames_folder, start_frame, end_frame, skip_factor, frame_offset, compression = args
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (current_frame - start_frame) % skip_factor == 0:
            frame_number = (current_frame - start_frame) // skip_factor + frame_offset
            frame_filename = os.path.join(output_frames_folder, f"{frame_number:07d}.png")
            
            os.makedirs(output_frames_folder, exist_ok=True)
            
            cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_PNG_COMPRESSION, compression])

        current_frame += 1

    cap.release()

def process_video(args):
    video_file, input_video_folder, output_frames_folder, target_fps, drop_start, drop_end, processes, compression = args
    
    video_path = os.path.join(input_video_folder, video_file)
    video_name = os.path.splitext(video_file)[0]
    output_frames_subfolders = os.path.join(output_frames_folder, video_name)
    os.makedirs(output_frames_subfolders, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(drop_start * original_fps)
    end_frame = total_frames - int(drop_end * original_fps)
    skip_factor = max(1, int(round(original_fps / target_fps)))

    cap.release()

    chunk_size = (end_frame - start_frame) // processes
    tasks = [(video_path, output_frames_subfolders, start_frame + i * chunk_size, min(end_frame, start_frame + (i + 1) * chunk_size), skip_factor, i * (chunk_size // skip_factor), compression) for i in range(processes)]

    with Pool(processes=processes) as pool:
        pool.map(process_chunk, tasks)

    logging.info(f"Video {video_name} processed.")

def process_all_videos():
    setup_logger()

    data_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(data_dir, "config.yaml")

    with open(config_path, 'r') as config_file:
        configs = yaml.safe_load(config_file)

    for config in configs:
        name = config['name']
        input_video_folder = os.path.join(data_dir, "videos", name)
        output_frames_folder = os.path.join(data_dir, "frames", name)
        target_fps = config.get('target_fps', 5)
        drop_start = config.get('drop_start', 0)
        drop_end = config.get('drop_end', 0)
        processes = config.get('processes', cpu_count())
        compression = config.get('compression', 0)

        video_files = [f for f in os.listdir(input_video_folder) if f.lower().endswith('.mp4')]
        args_list = [(video_file, input_video_folder, output_frames_folder, target_fps, drop_start, drop_end, processes, compression) for video_file in video_files]

        start_time = time.time()
        for args in args_list:
            process_video(args)
        end_time = time.time()

        logging.info(f"Total videos processed in '{input_video_folder}': {len(video_files)}. Total time: {end_time - start_time:.2f} seconds.")

def main():
    process_all_videos()

if __name__ == "__main__":
    main()
