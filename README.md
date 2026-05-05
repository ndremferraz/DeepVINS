# DeepVINS

  Real-time pose estimation is a fundamental capability in embodied AI systems. Autonomous drones, mobile
robots, and AR/VR headsets must continuously estimate their motion in order to navigate and interact
reliably with their environments, especially when external positioning signals are unavailable.

  Visual-inertial odometry (VIO) addresses this problem by combining image streams with inertial measurements from an IMU. The visual modality provides geometric and appearance information from the
scene, while the inertial modality supplies robust, high-frequency motion cues. Together, these signals enable
estimation of six-degree-of-freedom pose, typically represented as translation and orientation relative to an
initial reference frame.

  DeepVINS is an attempt to create a light weight Visual-Inertial Fusion transformer
The model was trainned on the EurocMav Dataset and due to I/O and storage limitations the entire model was trained entirely on Google Colab. 

- The Model Declaration, batching and trainning loop are contained in [DeepVINS Trainning Notebook]()

