# This is a sample Python script.
from extract_frames import extract_frames
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    extract_frames('video_left_camera/left_video.mp4', 'video_left_camera/extracted_frames')
    extract_frames('video_right_camera/right_video.mp4', 'video_right_camera/extracted_frames')
