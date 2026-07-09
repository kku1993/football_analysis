`/home/kevin/projects/football_analysis/editor_data/frames` contains frames extracted
from a video of a soccer game, recording sequentially.

`/home/kevin/projects/football_analysis/editor_data/annotations.json` contains the bounding
boxes for each player and ball in each frame, while
`/home/kevin/projects/football_analysis/output_videos/tracking_data.json` contains
the player and ball's positions in each frame translated to be relative to the pitch,
with the pitch center being (0, 0).

I want you to do this in order:

1) Identify and fix mis-labeled players. There are 2 cases:
  - a player might be labeled in frame A, disappear in frame A+1, and reappear in frame A+2 as a differently labeled player. In this case, we should label the player in A+1 and A+2 as the original play in frame A
  - a player might not be labeled in frame A, but labeled in frame A+1, we should try to label the player in A with the same label

2) Smooth movement
  Right now player movements are jittery due to camera calibation differences and bounding box inprecision.
  Try to smooth out each player's movement across frames.


Write all your fixed into a new file called
`/home/kevin/projects/football_analysis/output_videos/tracking_data_fixed.json`

DO NOT modify other files
