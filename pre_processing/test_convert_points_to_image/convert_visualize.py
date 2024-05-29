#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import numpy as np
import vispy
import imageio
import math
from vispy.scene import visuals, SceneCanvas
from matplotlib import pyplot as plt
from itertools import product

class LaserScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']
  def __init__(self, project=False, H=128, W=1024, fov_up=22.5, fov_down=-22.5):
  # def __init__(self, project=False, H=260, W=1840, fov_up=22.5, fov_down=-22.5):
    self.project = project
    self.proj_H = H
    self.proj_W = W
    self.proj_fov_up = fov_up
    self.proj_fov_down = fov_down
    self.reset()
  def reset(self):
    """ Reset scan members. """
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission
    # projected range image - [H,W] range (-1 is no data)
    self.proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
    # unprojected range (list of depths for each point)
    self.unproj_range = np.zeros((0, 1), dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)

    # projected remission - [H,W] intensity (-1 is no data)
    self.proj_remission = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    self.proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)

    # for each point, where it is in the range image
    self.proj_x = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: x
    self.proj_y = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    self.proj_mask = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)       # [H,W] mask
  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]

  def __len__(self):
    return self.size()

  def open_scan(self, filename):
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()

    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, " "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    # if all goes well, open pointcloud
    scan = np.fromfile(filename, dtype=np.float32)
    # scan = scan.reshape((-1, 4))
    scan = scan.reshape((-1, 8))
    # put in attribute
    points = scan[:, 0:3]    # get xyz
    remissions = scan[:, 4]
    self.set_points(points, remissions)

  def set_points(self, points, remissions=None):
    """ Set scan attributes (instead of opening from file)
    """
    # reset just in case there was an open structure
    self.reset()

    # check scan makes sense
    if not isinstance(points, np.ndarray):
      raise TypeError("Scan should be numpy array")

    # check remission makes sense
    if remissions is not None and not isinstance(remissions, np.ndarray):
      raise TypeError("Remissions should be numpy array")

    # put in attribute
    self.points = points    # get xyz
    if remissions is not None:
      self.remissions = remissions  # get remission
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

    # if projection is wanted, then do it and fill in the structure
    if self.project:
      self.do_range_projection()

  def do_range_projection(self):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    
    # laser parameters
    fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(self.points, 2, axis=1)
    # get scan components
    scan_x = self.points[:, 0]
    scan_y = self.points[:, 1]
    scan_z = self.points[:, 2]
    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= self.proj_W                              # in [0.0, W]
    proj_y *= self.proj_H                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x).astype(np.int32)
    proj_x = np.clip(proj_x, 0, self.proj_W - 1)   # in [0, W-1]

    proj_y = np.floor(proj_y).astype(np.int32)
    proj_y = np.clip(proj_y, 0, self.proj_H - 1)   # in [0, H-1]

    self.proj_x = np.copy(proj_x)  # store a copy in original order
    self.proj_y = np.copy(proj_y)  # store a copy in original order

    # copy of depth in original order
    self.unproj_range = np.copy(depth)

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(self.proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    self.proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(self.proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    self.proj_y = np.copy(proj_y)  # stope a copy in original order

    # copy of depth in original order
    self.unproj_range = np.copy(depth)

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = self.points[order]
    remission = self.remissions[order]
    depth = remission
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # # Creating a list of pairs [x, y], setting x_val and y_val to 0 when remission is 0
    # pairs = []
    # for ind, value in enumerate(remission):
    #     if value != 0:
    #         x_val, y_val = proj_x[ind], proj_y[ind]
    #     else:
    #         x_val, y_val = 0, 0
    #     pairs.append([x_val, y_val+70])

    # # Writing the pairs to a text file in the order of remission
    # with open('output.txt', 'w') as file:
    #     for x_val, y_val in pairs:
    #         file.write(f"{x_val} {y_val}\n")

    # # Creating a list of pairs [x, y], setting x_val and y_val to 0 when remission is 0
    # pairs = []
    # for ind, value in enumerate(remission):
    #     x_val, y_val = proj_x[ind], proj_y[ind]
    #     pairs.append([x_val-10, y_val+70])

    # # Writing the pairs to a text file in the order of remission
    # with open('output_2.txt', 'w') as file:
    #     for x_val, y_val in pairs:
    #         file.write(f"{x_val} {y_val}\n")


    # assing to images
    self.proj_range[proj_y, proj_x] = depth
    self.proj_xyz[proj_y, proj_x] = points
    self.proj_remission[proj_y, proj_x] = remission
    self.proj_idx[proj_y, proj_x] = indices
    self.proj_mask = (self.proj_idx > 0).astype(np.int32)

class LaserScanVis:
  """Class that creates and handles a visualizer for a pointcloud"""

  def __init__(self, scan, scan_names, offset=0):
    self.scan = scan
    self.scan_names = scan_names
    self.offset = offset    
    self.reset()
    self.update_scan()    
    
  def reset(self):
    """ Reset. """
    # last key press (it should have a mutex, but visualization is not
    # safety critical, so let's do things wrong)
    self.action = "no"  # no, next, back, quit are the possibilities

    # new canvas prepared for visualizing data
    self.canvas = SceneCanvas(keys='interactive', show=True)
    # interface (n next, b back, q quit, very simple)
    self.canvas.events.key_press.connect(self.key_press)
    self.canvas.events.draw.connect(self.draw)
    # grid
    self.grid = self.canvas.central_widget.add_grid()

    # laserscan part
    self.scan_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.canvas.scene)
    self.grid.add_widget(self.scan_view, 0, 0)
    self.scan_vis = visuals.Markers()
    self.scan_view.camera = 'turntable'
    self.scan_view.add(self.scan_vis)
    visuals.XYZAxis(parent=self.scan_view.scene)

    # self.red_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.canvas.scene)
    # self.grid.add_widget(self.red_view, 0, 0)
    # self.red_vis = visuals.Markers()
    # self.red_view.camera = 'turntable'
    # self.red_view.add(self.red_vis)
    # visuals.XYZAxis(parent=self.red_view.scene)

    # self.green_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.canvas.scene)
    # self.grid.add_widget(self.green_view, 0, 1)
    # self.green_vis = visuals.Markers()
    # self.green_view.camera = 'turntable'
    # self.green_view.add(self.green_vis)
    # visuals.XYZAxis(parent=self.green_view.scene)

    # self.blue_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.canvas.scene)
    # self.grid.add_widget(self.blue_view, 0, 2)
    # self.blue_vis = visuals.Markers()
    # self.blue_view.camera = 'turntable'
    # self.blue_view.add(self.blue_vis)
    # visuals.XYZAxis(parent=self.blue_view.scene)

    # self.add_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.canvas.scene)
    # self.grid.add_widget(self.add_view, 0, 3)
    # self.add_vis = visuals.Markers()
    # self.add_view.camera = 'turntable'
    # self.add_view.add(self.add_vis)
    # visuals.XYZAxis(parent=self.add_view.scene)

    # img canvas size
    self.multiplier = 1
    self.canvas_W = 1024
    self.canvas_H = 128

    # self.canvas_W = 1840
    # self.canvas_H = 230

    # new canvas for img
    self.img_canvas = SceneCanvas(keys='interactive', show=True, size=(self.canvas_W, self.canvas_H * self.multiplier))
    # grid
    self.img_grid = self.img_canvas.central_widget.add_grid()
    # interface (n next, b back, q quit, very simple)
    self.img_canvas.events.key_press.connect(self.key_press)
    self.img_canvas.events.draw.connect(self.draw)

    # add a view for the depth
    self.img_view = vispy.scene.widgets.ViewBox(parent=self.img_canvas.scene)
    self.img_grid.add_widget(self.img_view, 0, 0)
    self.img_vis = visuals.Image(cmap='viridis')
    self.img_view.add(self.img_vis)

  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

  def update_scan(self):
    # first open data
    self.scan.open_scan(self.scan_names[self.offset])

    # then change names
    title = "scan " + str(self.offset) + " of " + str(len(self.scan_names))
    self.canvas.title = title
    self.img_canvas.title = title

    # then do all the point cloud stuff    
    # plot scan
    power = 4
    range_data = np.copy(self.scan.unproj_range)
    
    range_data = range_data**(1 / power)
    viridis_range = ((range_data - range_data.min()) / (range_data.max() - range_data.min()) * 255).astype(np.uint8)
    viridis_map = self.get_mpl_colormap("viridis")
    viridis_colors = viridis_map[viridis_range]
    self.scan_vis.set_data(self.scan.points,
                           face_color=viridis_colors[..., ::-1],
                           edge_color=viridis_colors[..., ::-1],
                           size=1)

    # # Assuming you have a function to retrieve the "Blues" colormap, similar to get_mpl_colormap
    # reds_map = self.get_mpl_colormap("Reds")
    # with open('rgb_2.txt', 'r') as file:
    #     rgb_range_values = [tuple(map(int, line.split())) for line in file]
    # reds_range = [row[0] for row in rgb_range_values]
    # reds_range = np.array(reds_range)
    # reds_colors = reds_map[reds_range]
    # self.red_vis.set_data(self.scan.points,
    #                       face_color=reds_colors[..., ::-1],  # Reverse color channels if needed
    #                       edge_color=reds_colors[..., ::-1],
    #                       size=1)
    
    # greens_map = self.get_mpl_colormap("Greens")
    # with open('rgb_2.txt', 'r') as file:
    #     rgb_range_values = [tuple(map(int, line.split())) for line in file]
    # greens_range = [row[1] for row in rgb_range_values]
    # greens_range = np.array(greens_range)
    # greens_colors = greens_map[greens_range]
    # self.green_vis.set_data(self.scan.points,
    #                       face_color=greens_colors[..., ::-1],  # Reverse color channels if needed
    #                       edge_color=greens_colors[..., ::-1],
    #                       size=1)

    # blues_map = self.get_mpl_colormap("Blues")
    # with open('rgb_2.txt', 'r') as file:
    #     rgb_range_values = [tuple(map(int, line.split())) for line in file]
    # blues_range = [row[2] for row in rgb_range_values]
    # blues_range = np.array(blues_range)
    # blues_colors = blues_map[blues_range]
    # self.blue_vis.set_data(self.scan.points,
    #                       face_color=blues_colors[..., ::-1],  # Reverse color channels if needed
    #                       edge_color=blues_colors[..., ::-1],
    #                       size=1)

    # # Assuming white color is represented as (1, 1, 1) in RGB format
    # white_color = (0.5, 0.5, 0.5)
    # self.scan_vis.set_data(self.scan.points,
    #                       face_color=white_color,
    #                       edge_color=white_color,
    #                       size=1)

    # # Read RGB values from the text file
    # with open('rgb_2.txt', 'r') as file:
    #     rgb_values = [tuple(map(int, line.split())) for line in file]

    # # Convert the RGB values to a numpy array and normalize to [0, 1]
    # rgb_values_array = np.array(rgb_values)
    # with open('rgb_3.txt', 'w') as output_file:
    #     for rgb in zip(rgb_values_array):
    #         # Convert the RGB values to the desired format
    #         rgb_str = " ".join(map(str, rgb))
    #         output_file.write(f"{rgb_str}\n")

    # # Set RGB values to each point in self.scan.points
    # self.scan_vis.set_data(self.scan.points,
    #                       face_color=rgb_values_array,
    #                       edge_color=rgb_values_array,
    #                       size=1)
    # plot range imageintensity
    data = np.copy(self.scan.proj_range)
    data[data > 0] = data[data > 0]**(1 / power)
    data[data < 0] = data[data > 0].min()
    data = (data - data[data > 0].min()) / (data.max() - data[data > 0].min())
    self.img_vis.set_data(data)
    self.img_vis.update()
    
    image = self.img_canvas.render()
    imageio.imwrite('/media/khanh/ubuntu/PAPER_IV_2024/code/pre_processing/convert_points_to_image/'+'{0:010d}.png'.format(self.offset), image)
  # interface
  def key_press(self, event):
    
    self.canvas.events.key_press.block()
    self.img_canvas.events.key_press.block()
    if event.key == 'N':
      self.offset += 1     
      self.update_scan()

    elif event.key == 'B':
      self.offset -= 1
      self.update_scan()
    elif event.key == 'Q' or event.key == 'Escape':
      self.destroy()

  def draw(self, event):
    if self.canvas.events.key_press.blocked():
      self.canvas.events.key_press.unblock()
    if self.img_canvas.events.key_press.blocked():
      self.img_canvas.events.key_press.unblock()

  def destroy(self):
    # destroy the visualization
    self.canvas.close()
    self.img_canvas.close()
    vispy.app.quit()

  def run(self):
    vispy.app.run()

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser("./convert_visualize.py")
  parser.add_argument(
      '--sequence', '-s',
      type=str,
      required=False,
      help='Sequence to visualize. Defaults to %(default)s',
  )
  parser.add_argument(
      '--offset',
      type=int,
      default=0,
      required=False,
      help='Sequence to start. Defaults to %(default)s',
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Sequence", FLAGS.sequence)
  print("offset", FLAGS.offset)
  print("*" * 80)

  # does sequence folder exist?
  scan_paths = os.path.join(FLAGS.sequence)
  if os.path.isdir(scan_paths):
    print("Sequence folder exists! Using sequence from %s" % scan_paths)
  else:
    print("Sequence folder doesn't exist! Exiting...")
    quit()

  # populate the pointclouds
  scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_paths)) for f in fn]
  scan_names.sort()

  # create a scan
  scan = LaserScan(project=True)  
  vis = LaserScanVis(scan=scan, 
                     scan_names=scan_names, 
                     offset=FLAGS.offset)
  
  print("To navigate:")
  print("\tb: back (previous scan)")
  print("\tn: next (next scan)")
  print("\tq: quit (exit program)")
  # run the visualizer
  vis.run()