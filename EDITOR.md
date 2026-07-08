# Editor

The editor is an interactive app that lets a human user correct
the the inferred tracking data by updating bounding boxes on images.

## Definition

See tracking-schema.json for information we need

- Player bounding box - identifies a player on the pitch. The box should include information about the coordinates of the player, the team of the player (offence or defence), and a unique tracking id
- Ball bounding box - identifies the ball on the pitch. the box should include information about the coordinates of the ball, its state: Active, Goal, or OutOfBounds, and whether it's being kicked or received by a player.

## Workflow

1. main.py generates tracking data and creates one image per frame, with
   bounding boxes drawn around ball and all players

2. Allow human user to go through each frame and add/remove/edit bounding boxes.
   Human should be able to create bounding box by dragging it on the image, and
   human should be allowed to enter/edit the metadata associated with each box.

3. To assist with editing, show the frame immediately before and after the current
   frame being edited, along with all the bounding boxes on the before/after frames.
