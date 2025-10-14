import os
import subprocess
import shutil

def merge_multiple_videos(filepath: str, output_path: str = None) -> str:
    dirname, filename = os.path.split(filepath)
    base, ext = os.path.splitext(filename)
    verbose = True
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"First segment not found: {filepath}")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Install it or add to PATH.")

    files = [os.path.join(dirname, filename)]

    # Pattern 1: GoPro 'GX01xxxx.ext' → 'GX02xxxx.ext', ...
    if base.startswith("GX") and len(base) >= 6:
        base_suffix = base[4:]  # keep part after 'GX01'
        for i in range(2, 100):  # allow many segments
            continuation_name = f"GX0{i}{base_suffix}{ext}"
            candidate_path = os.path.join(dirname, continuation_name)
            if os.path.exists(candidate_path):
                files.append(candidate_path)
            else:
                break
    else:
        # Pattern 2: '<base>.ext' → '<base>_2.ext', '<base>_3.ext', ...
        for i in range(2, 100):
            continuation_name = f"{base}_{i}{ext}"
            candidate_path = os.path.join(dirname, continuation_name)
            if os.path.exists(candidate_path):
                files.append(candidate_path)
            else:
                break

    if verbose:
        print("[merge] Found segments:")
        for p in files:
            print("  -", os.path.abspath(p))

    # If there are no continuation files, return original path (but tell the user)
    if len(files) == 1:
        if verbose:
            print("[merge] No continuation segments found. Returning original file.")
        return os.path.abspath(files[0])

    abs_paths = [os.path.abspath(p).replace("\\", "/") for p in files]
    list_file_path = os.path.join(dirname, "concat_list.txt")

    # Build output path
    if output_path is None or os.path.isdir(output_path):
        out_dir = dirname if output_path is None else output_path
        output_path = os.path.join(out_dir, f"{base}_merged{ext}")
    output_path = os.path.abspath(output_path).replace("\\", "/")

    # Write concat list
    with open(list_file_path, "w", encoding="utf-8") as f:
        for path in abs_paths:
            f.write(f"file '{path}'\n")

    # Run ffmpeg and capture errors
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file_path,
        "-c", "copy",
        output_path
    ]
    if verbose:
        print("[merge] Running:", " ".join(cmd))

    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed ({result.returncode}):\n{result.stderr}")
    finally:
        try:
            os.remove(list_file_path)
        except OSError:
            pass

    return output_path

if __name__ == "__main__":
    out = merge_multiple_videos(
        r"C:\Users\pimha\PycharmProjects\photogrammetry_thesis\Photogrammetry\input\right_videos\GX010353.MP4"
    )
    print("RESULT:", out)

