"""Helpers for deriving Gazebo camera plugin topics from a ROS image topic."""

DEFAULT_CAMERA_TOPIC = '/vns_drone/downward_camera/image_raw'

_IMAGE_SUFFIX = '/image_raw'
_NAMESPACE_PLACEHOLDER = '<namespace>/vns_drone</namespace>'
_CAMERA_NAME_PLACEHOLDER = '<camera_name>downward_camera</camera_name>'
_CAMERA_PLUGIN_START = "<plugin name='camera_controller' filename='libgazebo_ros_camera.so'>"
_PLUGIN_END = '</plugin>'


def split_camera_topic(topic: str) -> tuple[str, str]:
    """Split a ROS image topic into gazebo_ros camera namespace and name."""
    if not topic.endswith(_IMAGE_SUFFIX):
        raise ValueError("camera topic must end with '/image_raw'")

    base_topic = topic[: -len(_IMAGE_SUFFIX)]
    namespace, separator, camera_name = base_topic.rpartition('/')
    if not separator or not namespace or not camera_name:
        raise ValueError(
            "camera topic must include a namespace and camera name before '/image_raw'"
        )

    return namespace, camera_name


def render_sdf_with_camera_topic(sdf_text: str, topic: str) -> str:
    """Render the camera plugin topic placeholders in the drone SDF."""
    if topic == DEFAULT_CAMERA_TOPIC:
        return sdf_text

    namespace, camera_name = split_camera_topic(topic)
    plugin_start = sdf_text.find(_CAMERA_PLUGIN_START)
    if plugin_start == -1:
        raise ValueError('camera plugin block not found in SDF')

    plugin_end = sdf_text.find(_PLUGIN_END, plugin_start)
    if plugin_end == -1:
        raise ValueError('camera plugin block is missing its closing tag in SDF')
    plugin_end += len(_PLUGIN_END)

    camera_plugin_block = sdf_text[plugin_start:plugin_end]

    if camera_plugin_block.count(_NAMESPACE_PLACEHOLDER) != 1:
        raise ValueError('camera namespace placeholder not found exactly once in SDF')
    if camera_plugin_block.count(_CAMERA_NAME_PLACEHOLDER) != 1:
        raise ValueError('camera name placeholder not found exactly once in SDF')

    rendered_camera_plugin = (
        camera_plugin_block.replace(
            _NAMESPACE_PLACEHOLDER, f'<namespace>{namespace}</namespace>'
        ).replace(
            _CAMERA_NAME_PLACEHOLDER,
            f'<camera_name>{camera_name}</camera_name>',
        )
    )

    return (
        sdf_text[:plugin_start]
        + rendered_camera_plugin
        + sdf_text[plugin_end:]
    )
