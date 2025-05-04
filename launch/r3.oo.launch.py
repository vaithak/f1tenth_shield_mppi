import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
   config = os.path.join(
      get_package_share_directory('shield_mppi'),
      'config',
      'r3.params.yaml'
      )

   return LaunchDescription([
      Node(
         package='shield_mppi',
         executable='opponent_detection_node_cpp',
         namespace='',
         name='opponent_detection',
         parameters=[config]
      )
      ,Node(
         package='shield_mppi',
         executable='opponent_tracking.py',
         namespace='',
         name='opponent_tracking',
         parameters=[config]
      )
   ])