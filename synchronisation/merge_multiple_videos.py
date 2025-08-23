import os

import subprocess

def merge_two_videos(filepath, output_path=None):
    dirname, filename = os.path.split(filepath)
    base, ext = os.path.splitext(filename)

    # Always include the original file
    files = [os.path.join(dirname, filename)]

    if base.startswith("GX") and len(base) >= 6:
        base_suffix = base[5:]
        for i in range(2, 10):
            continuation_name = f"GX0{i}{base_suffix}{ext}"
            candidate_path = os.path.join(dirname, continuation_name)
            if os.path.exists(candidate_path):
                files.append(candidate_path)
            else:
                break
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

    # Otherwise, run ffmpeg to merge
    abs_paths = [os.path.abspath(p).replace("\\", "/") for p in files]
    list_file_path = os.path.join(dirname, "concat_list.txt")
    with open(list_file_path, "w") as f:
        for path in abs_paths:
            f.write(f"file '{path}'\n")

    if output_path is None:
        output_path = os.path.join(dirname, f"{base}_merged{ext}")
    output_path = output_path.replace("\\", "/")

    cmd = [
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", list_file_path,
        "-c", "copy",
        output_path
    ]

    subprocess.run(cmd, check=True)
    os.remove(list_file_path)
    return output_path