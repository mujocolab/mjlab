# Unitree D1

This D1 arm description was sourced from the public
`elijah-waichong-chan/hq-pcot` ROS 2 workspace, package `d1_description`.
The package metadata declares a BSD license.

The URDF is a SolidWorks export with mesh collision geometry and zero
effort/velocity limits. mjlab supplies conservative placeholder actuator
limits in `d1_constants.py`. The combined Go2+D1 asset currently disables D1
mesh collisions and uses the arm visually/kinematically; add simplified
primitive colliders before using it for contact-rich manipulation.
