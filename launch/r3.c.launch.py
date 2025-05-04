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
         executable='shield_mppi_node.py',
         namespace='',
         name='shield_mppi_node',
         parameters=[config]
      )
   ])