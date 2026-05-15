# DroneRL — Autonomous Obstacle Navigation RL

A Unity-based reinforcement learning environment where a physics-driven drone learns to navigate dense, procedurally generated forests to land on a dynamic target. Trained entirely using Proximal Policy Optimization (PPO) and Curriculum Learning.

## What It Does

Traditional pathfinding algorithms rely on pre-calculated nav-meshes. DronePilot actively perceives its environment using simulated sensors and physics to calculate its own continuous flight paths in real-time.

* **Continuous Flight Control:** The agent outputs continuous vector arrays mapping to four independent forces: X/Z horizontal thrust, Y-axis ascension, and Yaw rotation, mimicking true quadcopter physics.
* **Procedural Forest Generation:** Every episode spawns a completely randomized forest density and a dynamic target podium location, preventing the AI from memorizing static maps.
* **Raycast Perception:** The drone uses a 3D ray-perception sensor array to detect trees and boundaries at high speeds, feeding distance metrics directly into the neural network.
* **Curriculum Scaling:** The environment dynamically adjusts the distance of the target podium based on the drone's current success rate, ensuring it stays in the optimal learning zone.

## Progression & Training History

The following progression outlines the major hurdles and architectural changes across the training sessions.

### Experiments 1 & 2
Initial tests utilized a smaller neural network (128 hidden units, 2 layers). The drone struggled to comprehend the continuous application of yaw alongside forward thrust, resulting in a localized minima where it simply spun in circles to avoid trees, ultimately timing out.

<p align="center">
  <img src="GIFs/DroneRLTest1.gif" width="45%" alt="Test 1 Spinning" />
  <img src="GIFs/DroneRLTest2.gif" width="45%" alt="Test 2 Spinning" />
</p>

### Experiment 3
To solve the spinning and frequent crashes, I implemented two major changes were for Test 3:
1.  **Curriculum Learning:** The `LevelManager` was modified to spawn the podium within a strict 10-meter radius, gradually expanding outward only as the agent mastered the current distance.
2.  **Hidden Layer Expansion:** The network architecture was upgraded to **256 hidden units across 3 layers** to handle the complex spatial math.

<p align="center">
  <img src="GIFs/DroneRLTest3Good.gif" width="80%" alt="Test 3 Success" />
</p>

## Behaviour Problems
While Test 3 successfully reached the podium, the drone developed three highly specific, unintended behavioral habits to "game" the reward system:

<p align="center">
  <img src="GIFs/DroneRLTest3Backwards.gif" width="32%" alt="Moving Backwards" />
  <img src="GIFs/DroneRLTest3Rush.gif" width="32%" alt="Rushing" />
  <img src="GIFs/DroneRLTest3Stuck.gif" width="32%" alt="Ceiling Escape" />
</p>

1.  **Reverse Flight (Left):** The drone utilized negative Z-thrust to reach the target rather than rotating, effectively flying blind because its raycasts face forward.
2.  **Tree Rushing (Center):** The drone accelerated too fast into dense clusters and failed to brake in time, lacking the velocity-awareness to calculate stopping distances.
3.  **Accepting Fate (Right):** When overwhelmed by trees, the drone realized flying straight up into the height ceiling yielded a "cleaner" death than hitting a tree, so it simply stalled and ascended.

### Experiment 4: Reward Shaping (4 Updates)
To correct these behaviors, the `DroneAgent.cs` observation and reward logic was heavily restructured:
* **Dot Product Alignment:** Added `Vector3.Dot(transform.forward, dirToTarget)` to actively reward the drone for pointing its nose at the target, eliminating backward flight.
* **Angular & Linear Velocity Observations:** Added `rb.linearVelocity` and `rb.angularVelocity.y` directly into the sensor payload so the brain understands its own momentum.
* **The Anti-Stall Penalty:** Implemented a negative reward tick if velocity drops near zero while not at the target, forcing the drone to actively seek paths rather than stalling.
* **Soft Ceiling Penalties:** Replaced the hard kill-box with a soft penalty zone at 80% of the maximum height to discourage vertical escapes.

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Engine** | Unity 6.0+ |
| **RL Framework** | Unity ML-Agents 1.1.0 |
| **Algorithm** | PPO (Proximal Policy Optimization) |
| **Training Backend** | PyTorch (CPU-Optimized) |
| **Environment Logic** | C# |
| **Cloud Compute** | Google Cloud Platform (e2-standard-8 and n1-standard-4 VMs) |
| **Model Export** | ONNX |

## Machine Learning Components

### 1. Proximal Policy Optimization (PPO)
The agent is trained using OpenAI's PPO, an Actor-Critic on-policy algorithm. The "Actor" network decides the continuous thrust values, while the "Critic" network evaluates the value of the current spatial state. 

### 2. Multi-Modal Vector Observations
The drone does not use visual camera data (pixels). Instead, it processes a highly efficient vector array of floats:
* Normalized 3D direction to target
* Normalized scalar distance to target
* Current Forward/Right orientation vectors
* Linear and Angular velocity arrays
* RayPerception distances (Obstacle detection)

### 3. CPU-Optimized Inference
Because the environment is entirely vector-based, training bypassed GPU bottlenecks by utilizing a high-core-count GCP compute instance. The physics simulations and PPO gradients were calculated entirely on 8 CPU cores, allowing for massive batch processing without memory limits. The final `.onnx` model was compressed via ONNX C-API to a highly efficient 590KB inference engine. (Note that the first two experiments were done on GPU while experiment 3 was run on CPU)
