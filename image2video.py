import os
import moviepy.video.io.ImageSequenceClip
import numpy as np

image_folder = 'image2video'

# Generate the texture animation
print("Generating MP4 from FEA Textures.")
image_files = [os.path.join(image_folder,img)
               for img in os.listdir(image_folder)
               if img.endswith(".png")]
image_ids = np.array([int(i.split('.')[0][-3:]) for i in image_files])
sorted_ids = np.argsort(image_ids)
image_files = list(np.array(image_files)[sorted_ids])
print(image_files)
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=30)
clip.write_videofile('video.mp4')