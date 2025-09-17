"""This code can be used to merge split GoPro (and similar) video segments into a single file
using FFmpeg’s concat demuxer. It supports the GoPro naming scheme 'GX01… → GX02… → …'
as well as generic continuations like '<base>_2.mp4', '<base>_3.mp4'.

You can adapt the filename patterns, output naming, or FFmpeg flags to your dataset.
"""

import os
import subprocess


def merge_two_videos(filepath: str, output_path: str = None) -> str:
    """RELEVANT FUNCTION INPUTS:
    - filepath: path to the first (or only) video segment, e.g. '.../GX01abcd.mp4'
                or '.../session.mp4' when using the '<base>_2.mp4' pattern.
    - output_path: optional explicit path for the merged file. If None, the merged file
                   is written next to the inputs as '<base>_merged<ext>'.

    RETURNS:
    - Absolute path to the merged (or original, if no continuations found) video file.

    Behavior:
    - If 'GX01…' is detected, it searches for 'GX02…' to 'GX09…' in the same folder.
    - Otherwise, it searches for '<base>_2', '<base>_3', … in the same folder.
    - If no continuation segments exist, the original file path is returned.
    - If segments exist, an FFmpeg concat list is created and merged with stream copy.
    """
    dirname, filename = os.path.split(filepath)
    base, ext = os.path.splitext(filename)

    # Always include the original file
    files = [os.path.join(dirname, filename)]

    # Pattern 1: GoPro 'GX01xxxx.ext' → 'GX02xxxx.ext', ...
    if base.startswith("GX") and len(base) >= 6:
        base_suffix = base[5:]  # keep trailing part after 'GX0?'
        for i in range(2, 10):
            continuation_name = f"GX0{i}{base_suffix}{ext}"
            candidate_path = os.path.join(dirname, continuation_name)
            if os.path.exists(candidate_path):
                files.append(candidate_path)
            else:
                break
    # Pattern 2: '<base>.ext' → '<base>_2.ext', '<base>_3.ext', ...
    else:
        for i in range(2, 10):
            continuation_name = f"{base}_{i}{ext}"
            candidate_path = os.path.join(dirname, continuation_name)
            if os.path.exists(candidate_path):
                files.append(candidate_path)
            else:
                break

    # If there are no continuation files, return original path
    if len(files) == 1:
        return os.path.abspath(files[0])

    # Otherwise, merge via FFmpeg concat demuxer
    abs_paths = [os.path.abspath(p).replace("\\", "/") for p in files]
    list_file_path = os.path.join(dirname, "concat_list.txt")

    if output_path is None:
        output_path = os.path.join(dirname, f"{base}_merged{ext}")
    output_path = os.path.abspath(output_path).replace("\\", "/")

    # Write concat list and run ffmpeg; always clean up the list file
    try:
        with open(list_file_path, "w", encoding="utf-8") as f:
            for path in abs_paths:
                f.write(f"file '{path}'\n")

        cmd = [
            "ffmpeg", "-y",           # overwrite without prompt
            "-f", "concat", "-safe", "0",
            "-i", list_file_path,
            "-c", "copy",             # stream copy (fast, lossless)
            output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        try:
            os.remove(list_file_path)
        except OSError:
            pass

    return output_path


# Utilities module; no runnable demo here.
if __name__ == "__main__":
    pass
