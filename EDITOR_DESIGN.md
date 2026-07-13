# Editor

This is an editor to allow humans to collaborate with computer vision models to
label players and ball in broadcast video for football/soccer matches.

## Step 1: per frame bounding boxes

This repository already contains a pipeline for analysing video. See @main.py

Adapt this pipeline to create a new pipeline to do this:

1. User pass in path to a video file from the command line (e.g. `input_videos/eng-nor-49-offside-no-goal.mp4`)
  - User can optionally pass in an `fps` parameter to specify the frame rate for frame extraction. Default to 12.
2. The script runs the pipeline and produces a directory
  - Directory name is the base name of the video without extension (e.g. `eng-nor-49-offside-no-goal`)
  - The directory contains:

  ```
  fps.txt # fps used for this pipeline
  frames/ # image of invididual frames 
    frame_001.png
    frame_002.png
    ...
  bbox/ # json containing player and ball bounding boxes per frame
    frame_001_bbox.json
    frame_002_bbox.json
    ...
  ```

  The schema of the bounding box json is defined in @bbox-schema.json
  Note we don't need to track the referees.


## Step 2: editor

The editor should be a simple python backend + a static js frontend.

Workflow part 1:

1. User gives path to the output directory from step 1 above to the editor
2. User opens web page - the page should show all the frames
3. Allow user to use arrow key to look through all frames and mark which frames they want to process (call these "to process frames")
  - Allow user to come back and reselect different set of to process frames at any time
  - NOTE: user may select non-consecutive frames

Once user has selected or confirmed the to process frames, enter workflow part 2:

1. For each to process frame, user adds calibration points for homography
  - Copy the features of /home/kku/projects/football-pitch-calibration/server.py - I want the same homography editing experience
  - Allow user to jump between different to process frames
  - Make sure the homography data structure is suitable for automation in the future - we will have a model generate calibration points automatically
2. Once homography information is available on frame, allow user to label the frame
  - User should be able to move bounding boxes on player and ball around by dragging
  - Each bounding box should be labled with the player id (or "ball")
  - User should be able to edit ball and player attributes, per @tracking-schema.json
  - Make sure x,y,z coorindates can be manually edited
  - Translate, in real time, each boxing box's center pixel location to (x, y) using the homography information. Let user enter z-axis themselves.
  - Make sure to support ball state and kick attributes as well
  - Player attributes (except x,y,z) should propagate to all frames (before and after current frame) for the same player id.
  - Allow user to one click specify whether team A or team B is offence vs defence and propagate that change to all players
  - Deleting a player bounding box should remove the bounding boxes for the same id across all subsequent frames
3. Have an "isolation" button - clicking this will hide all bounding boxes except the currently selected box across "to process" frames.
4. Allow user to specify "correct" sets of offence and defence player ids
  - Flag frames that are missing or have extra player ids present in the frame and make it easy for user to navigate to them
  - The "correct" sets can have start frame and end frame - the set only applies to this range. By default, new sets are applied to all subsequent frames

## Output

Have an export button to download labeled data that conforms to @tracking-schema.json

Since "to process" frames might not be consecutive, extrapolate data to fill in frames. Treat
the first frame (lexicographical order) as the beginning, and extrapolate frames
between each subsequent frame.

For extrapolation, assume the following:
   a) ball stays with player who received it (or the closest player if in beginning) until it's kicked. Keep the ball around 0.3m-0.5m (randomize a little) ahead of the player with possession.
   b) ball passes should follow a linear line to the receiver in the (x, y) plane. If z-axis is involved, generate a parabola in the z-plane.
   c) player run in mostly straight line between frames with some noise in acceleration. Also try to implement the property that defensive players want to stay close to offensive player that is close to them or run towards the ball if it's nearby

## Autosave state

The application should autosave state after every change, including calibration points, calculated homography information, edited bounding boxes, and player/ball attributes.
Save the state in a file called `label-system-state.json` within the data directory.

## Test data

See input_videos/eng-nor-49-offside-no-goal.mp4

In the sample data, there should be 8 England players (white, offence) and 9 Norway players (red, defence) + 1 Norway goal keeper (green).

## Tech stack

- Use a simple python backend
- Server a vanilla js page
