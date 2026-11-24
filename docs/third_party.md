# Third-party code and dependencies

The embedded LeRobot dataset utilities retain Hugging Face's Apache-2.0 notices.
The shared utility derived from NVIDIA code retains its Apache-2.0 notice.
Wan model components and weight conversion/loading code are derived from the
Wan / DiffSynth implementation used by the policy. Existing attribution notices
are retained in source files.

The pretrained LAM backbone implementation under
`src/wing/latent_action/source_lam` originates from DreamDojo and retains its
bundled Apache-2.0 license. WING-LAM training uses this backbone; its pretrained
checkpoint is an external input rather than a redistributed weight file.

Simulator implementations (LIBERO, RoboTwin, RoboCasa), robot SDKs, datasets,
pretrained model weights, and their assets are not redistributed here. Install
them separately and follow their respective license and usage terms. The WING
source license does not override third-party notices or grant rights to external
datasets and model assets.
