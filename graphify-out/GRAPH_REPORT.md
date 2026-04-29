# Graph Report - .  (2026-04-26)

## Corpus Check
- 15 files · ~128,113 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 92 nodes · 147 edges · 9 communities detected
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 16 edges (avg confidence: 0.76)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_ReachPose Environment Core|ReachPose Environment Core]]
- [[_COMMUNITY_Environment Registration & Setup|Environment Registration & Setup]]
- [[_COMMUNITY_Documentation & Assets|Documentation & Assets]]
- [[_COMMUNITY_Control Architecture Patterns|Control Architecture Patterns]]
- [[_COMMUNITY_PickAndPlace Observation System|PickAndPlace Observation System]]
- [[_COMMUNITY_PickAndPlace Reward & Success Logic|PickAndPlace Reward & Success Logic]]
- [[_COMMUNITY_PickAndPlace Initialization|PickAndPlace Initialization]]
- [[_COMMUNITY_Test Suite|Test Suite]]
- [[_COMMUNITY_MuJoCo Runtime Errors|MuJoCo Runtime Errors]]

## God Nodes (most connected - your core abstractions)
1. `PickAndPlaceEnv` - 32 edges
2. `ReachPoseEnv` - 13 edges
3. `PickAndPlaceEnv Class` - 8 edges
4. `ReachPoseEnv Class` - 8 edges
5. `Project README` - 6 edges
6. `Environments Documentation` - 5 edges
7. `MuJoCo Model Visualizer` - 4 edges
8. `VS050 Arm with Robotiq 2F-85 Gripper` - 4 edges
9. `VS050-ReachPose-v0 Environment` - 4 edges
10. `VS050-PickAndPlace-v0 Environment` - 4 edges

## Surprising Connections (you probably didn't know these)
- `PickAndPlace Environment Preview` --conceptually_related_to--> `PickAndPlaceEnv Class`  [INFERRED]
  assets/pick_and_place.gif → src/vs050_mujoco/envs/pick_and_place_env.py
- `ReachPose Environment Preview` --conceptually_related_to--> `ReachPoseEnv Class`  [INFERRED]
  assets/reach_pose.gif → src/vs050_mujoco/envs/reach_pose_env.py
- `VS050 Arm with Robotiq 2F-85 Gripper` --conceptually_related_to--> `vs050_2f85.xml Model`  [INFERRED]
  assets/vs050_2f85.png → src/vs050_mujoco/models/README.md
- `Standalone Smoke Test for ReachPose` --semantically_similar_to--> `Pytest Suite for PickAndPlace`  [INFERRED] [semantically similar]
  test_env.py → tests/test_env.py
- `PickAndPlaceEnv Class` --conceptually_related_to--> `Robotiq 2F-85 Gripper`  [EXTRACTED]
  src/vs050_mujoco/envs/pick_and_place_env.py → README.md

## Hyperedges (group relationships)
- **Gymnasium Environment Registration Flow** — pkg_init, pick_and_place_env, reach_pose_env [EXTRACTED 1.00]
- **MuJoCo Simulation Asset Pipeline** — visualize_module, scene_reach_xml, pick_and_place_scene_xml, vs050_xml, vs050_2f85_xml [EXTRACTED 1.00]
- **RL Environment Design Pattern** — pick_and_place_env, reach_pose_env, mujocoenv_base [INFERRED 0.85]

## Communities

### Community 0 - "ReachPose Environment Core"
Cohesion: 0.23
Nodes (5): MujocoEnv, Get MuJoCo ID or raise ValueError., VS050 reach-pose environment using MujocoEnv.      Observation (flat Box):, ReachPoseEnv, test_env()

### Community 1 - "Environment Registration & Setup"
Cohesion: 0.22
Nodes (14): Agent Notes for vs050-mujoco, Envs Package Init, Modular Over Monolithic Coding Style, gymnasium.envs.mujoco.MujocoEnv, PickAndPlaceEnv Class, pick_and_place_scene.xml Model, vs050_mujoco Package Init, ReachPoseEnv Class (+6 more)

### Community 2 - "Documentation & Assets"
Cohesion: 0.26
Nodes (12): DENSO VS050 6-DoF Robot Arm, Environments Documentation, Models Documentation, PickAndPlace Environment Preview, ReachPose Environment Preview, Project README, Robotiq 2F-85 Gripper, Standalone Smoke Test for ReachPose (+4 more)

### Community 3 - "Control Architecture Patterns"
Cohesion: 0.2
Nodes (3): vs050_mujoco – Gymnasium environments for the DENSO VS050 robot arm., Pick-and-Place Gymnasium Environment for DENSO VS050 + Robotiq 2F-85.  Refactore, ReachPose Gymnasium Environment for DENSO VS050 robot arm.  Refactored to inheri

### Community 4 - "PickAndPlace Observation System"
Cohesion: 0.33
Nodes (2): PickAndPlaceEnv, Pick-and-place task with the DENSO VS050 arm and Robotiq 2F-85 gripper.      **O

### Community 5 - "PickAndPlace Reward & Success Logic"
Cohesion: 0.32
Nodes (1): True if object is within _SUCCESS_DIST of target.

### Community 6 - "PickAndPlace Initialization"
Cohesion: 0.29
Nodes (1): Get MuJoCo ID or raise ValueError.

### Community 7 - "Test Suite"
Cohesion: 0.5
Nodes (2): Smoke-test for VS050 Pick-and-Place environment. Run from the repo root:     pyt, test_basic()

### Community 11 - "MuJoCo Runtime Errors"
Cohesion: 1.0
Nodes (2): MuJoCo Runtime Log, OpenGL Error 0x502 in mjr_makeContext

## Knowledge Gaps
- **12 isolated node(s):** `Smoke-test for VS050 Pick-and-Place environment. Run from the repo root:     pyt`, `Pick-and-Place Gymnasium Environment for DENSO VS050 + Robotiq 2F-85.  Refactore`, `Pick-and-place task with the DENSO VS050 arm and Robotiq 2F-85 gripper.      **O`, `Get MuJoCo ID or raise ValueError.`, `True if object is within _SUCCESS_DIST of target.` (+7 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `PickAndPlace Observation System`** (11 nodes): `PickAndPlaceEnv`, `._get_gripper_obs()`, `._get_joint_obs()`, `._get_object_obs()`, `._get_obs()`, `._get_target_obs()`, `._reset_arm()`, `._reset_gripper()`, `.reset_model()`, `._reset_simulation()`, `Pick-and-place task with the DENSO VS050 arm and Robotiq 2F-85 gripper.      **O`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `PickAndPlace Reward & Success Logic`** (8 nodes): `._build_step_return()`, `._check_grasp()`, `._check_success()`, `._compute_reward()`, `._get_obj_pos()`, `._get_pinch_pos()`, `._get_reset_info()`, `True if object is within _SUCCESS_DIST of target.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `PickAndPlace Initialization`** (7 nodes): `._get_id()`, `.__init__()`, `._init_cache_ids()`, `._init_ezpickle()`, `._init_mujoco_env()`, `._init_state()`, `Get MuJoCo ID or raise ValueError.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Suite`** (4 nodes): `Smoke-test for VS050 Pick-and-Place environment. Run from the repo root:     pyt`, `test_basic()`, `test_rgb_array()`, `test_env.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `MuJoCo Runtime Errors`** (2 nodes): `MuJoCo Runtime Log`, `OpenGL Error 0x502 in mjr_makeContext`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PickAndPlaceEnv` connect `PickAndPlace Observation System` to `ReachPose Environment Core`, `Control Architecture Patterns`, `PickAndPlace Reward & Success Logic`, `PickAndPlace Initialization`, `PickAndPlace Object Placement`, `PickAndPlace Step & Control`?**
  _High betweenness centrality (0.335) - this node is a cross-community bridge._
- **Why does `ReachPoseEnv` connect `ReachPose Environment Core` to `Control Architecture Patterns`?**
  _High betweenness centrality (0.215) - this node is a cross-community bridge._
- **Why does `vs050_mujoco – Gymnasium environments for the DENSO VS050 robot arm.` connect `Control Architecture Patterns` to `ReachPose Environment Core`, `PickAndPlace Observation System`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `PickAndPlaceEnv Class` (e.g. with `ReachPoseEnv Class` and `PickAndPlace Environment Preview`) actually correct?**
  _`PickAndPlaceEnv Class` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `ReachPoseEnv Class` (e.g. with `PickAndPlaceEnv Class` and `ReachPose Environment Preview`) actually correct?**
  _`ReachPoseEnv Class` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Smoke-test for VS050 Pick-and-Place environment. Run from the repo root:     pyt`, `Pick-and-Place Gymnasium Environment for DENSO VS050 + Robotiq 2F-85.  Refactore`, `Pick-and-place task with the DENSO VS050 arm and Robotiq 2F-85 gripper.      **O` to the rest of the system?**
  _12 weakly-connected nodes found - possible documentation gaps or missing edges._