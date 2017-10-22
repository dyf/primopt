import moviepy.editor as mpy
from scipy.misc import imread
import argparse
import glob

def framevid(pattern, fps, output):
    file_names = glob.glob(pattern)
    animation = mpy.ImageSequenceClip(file_names, fps=fps)
    animation.write_videofile(output, fps=fps)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pattern')
    parser.add_argument('fps',type=float)
    parser.add_argument('output')

    args = parser.parse_args()
    framevid(args.pattern, args.fps, args.output)

if __name__ == "__main__": main()
    