#!/usr/bin/env python3
"""GPS gate node for GNSS-denied simulation scenarios.

Relays /vns_drone/gps_raw -> /vns_drone/gps when GPS is enabled.
Toggle at runtime:
    ros2 service call /vns/set_gps_enabled std_srvs/srv/SetBool "{data: false}"
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_srvs.srv import SetBool


class GpsGateNode(Node):

    def __init__(self):
        super().__init__('gps_gate')
        self.declare_parameter('gps_enabled', True)
        self._enabled = self.get_parameter('gps_enabled').value

        self._sub = self.create_subscription(
            NavSatFix, '/vns_drone/gps_raw', self._on_gps, 10)
        self._pub = self.create_publisher(NavSatFix, '/vns_drone/gps', 10)
        self._srv = self.create_service(
            SetBool, '/vns/set_gps_enabled', self._toggle_cb)

        state = 'enabled' if self._enabled else 'DENIED'
        self.get_logger().info(f'GPS gate ready — GPS {state}')

    def _on_gps(self, msg: NavSatFix):
        if self._enabled:
            self._pub.publish(msg)

    def _toggle_cb(self, request, response):
        self._enabled = request.data
        state = 'enabled' if self._enabled else 'DENIED'
        self.get_logger().warn(f'GPS {state}')
        response.success = True
        response.message = f'GPS {state}'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GpsGateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
