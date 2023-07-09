# Introduction of RHSEDB
All data files of RHSEDB are '.npz' created by Numpy.<br>
The structure of data is shown as the follow:
```
'name': (1) char name of chinese character,
'stroke_name': (N) stroke names of this chinese character,
'stroke_label': (N) stroke label of this chinese character(same use wtih 'stroke_name', but labeled as number),
'reference_color_image': (3, 256, 256) reference Kaiti image, different strokes are marked by different colors,
'reference_single_image': (N, 256, 256) single stroke image of reference,
'reference_single_centroid': (N, 2) centroid of single stroke of reference,
'target_image': (1, 256, 256) target image,
'target_single_image': (N ,256, 256) single stroke image of target.
```
 