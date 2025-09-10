# VLA (Visual-Language Action) Humanoid Robot Control Project

## Project Overview

This is a Visual-Language Action (VLA) model-based humanoid robot control project, focusing on solving key problems in **26-DOF humanoid robot** whole-body motion prediction and execution.

Although current training data primarily focuses on upper body actions like waving and fist-clenching, the model architecture supports coordinated control of all 26 joints, including complete humanoid robot movements for legs, waist, arms, etc.

### Core Issues
- **Mid-state Prediction Decay**: 95% action amplitude decay when predicting from intermediate states
- **Trajectory Connection Jerk**: Uneven connections between adjacent trajectory segments, causing "jerky" motion
- **Incomplete Action Understanding**: Model lacks contextual understanding, unable to execute complete actions properly

### Robot Configuration
- **Joint Count**: 26 degrees of freedom (complete humanoid robot)
- **Joint Distribution**: 
  - Leg Joints: 0-11 (12 joints)
  - Left Arm Joints: 12-18 (7 joints) 
  - Right Arm Joints: 19-25 (7 joints)
- **Key Joints**: l_arm_pitch(12), l_arm_roll(13), l_arm_yaw(14)
- **Action Commands**: Whole-body actions including waving, fist-clenching

## Demo Video

<div align="center">
  <img src="video/wave.gif" alt="VLA Humanoid Robot Wave Demo" width="640">
</div>

*GIF showing the humanoid robot executing wave actions but still need to upgrade*


## Project Structure

```
vla/
├── simple_act_train.py          # Main training script (improved model)
├── smooth_act_inference_node.py # Inference node
├── improved_act_model.py        # Improved model architecture
├── trajectories/                # Training data
│   ├── wave_001.json ~ wave_008.json    # Wave action data
│   └── welcome_001.json ~ welcome_008.json # Fist-clenching action data
├── checkpoints/                 # Model save directory
│   ├── improved_act_model_v1/           # Improved model
│   └── key_joint_act_model_v1/          # Original model
└── modules/                      # Core modules
    ├── direct_action_predictor.py
    └── training_logic.py
```

## Core Technology

### Improved Model Architecture (`ImprovedKeyJointACTGenerator`)

1. **Action Intensity Regulator**
   - Solves mid-state prediction decay problem
   - Ensures complete action amplitude from any prediction state

2. **State Awareness Enhancer**
   - Enables model to understand current state and context
   - Avoids simple action repetition, understands action start, continuation, and end

3. **Enhanced Temporal Encoding**
   - Multi-scale temporal encoding (linear, sine, phase, frequency)
   - Better understanding of action temporal characteristics

4. **Multi-scale Prediction**
   - 3 parallel prediction heads ensure stable output
   - Reduces prediction variance, improves trajectory smoothness

### Key Features

- **Two-layer Architecture**: Instruction classification + action prediction
- **Key Joint Focus**: Automatic identification of key joints for each instruction
- **Temporal Smoothing**: Transformer processing for temporal dependencies
- **Instruction Diversity**: Ensures different instructions produce completely different action patterns

## Quick Start

### 1. Train Model

```bash
# Train improved model (recommended)
python3 simple_act_train.py

# Model will be saved to checkpoints/improved_act_model_v1/
```

### 2. Inference Testing

```bash
# Launch inference node
python3 smooth_act_inference_node.py
```

### 3. Model Validation

```bash
# Validate model performance
python3 model_vs_robot_comparison.py (this file has been deleted as mentioned, please create if needed)
```

## Training Configuration

### Model Parameters
- **State Dimension**: 26 (humanoid robot full body joint count)
- **Action Dimension**: 26 (humanoid robot full body joint count)
- **Instruction Count**: 2 (wave, fist-clenching and other whole-body actions)
- **Hidden Dimension**: 256
- **Trajectory Length**: 32 (predict next 32 steps)
- **Key Joint Count**: 8 (each instruction focuses on 8 key joints)

### Training Parameters
- **Batch Size**: 64
- **Learning Rate**: 1e-4
- **Training Epochs**: 2000
- **Optimizer**: AdamW
- **Loss Function**: Multi-objective combined loss

## Expected Results

| Issue | Original Model | Improved Model |
|-------|---------------|----------------|
| Fist-clenching Stuck | ❌ Stuck in middle | ✅ Complete execution |
| Jerky Motion | ❌ Connection stutter | ✅ Smooth and coherent |
| Action Understanding | ❌ Repeated waving | ✅ Context-aware |
| Mid-state Decay | ❌ 95% decay | ✅ Maintains amplitude |

## Key Improvements

### 1. Solving Mid-state Prediction Decay
```python
# Action intensity regulator
action_intensity = self.action_intensity_regulator(intensity_input)
final_predictions = final_predictions * action_intensity.unsqueeze(-1).unsqueeze(-1)
```

### 2. State Awareness Enhancement
```python
# State awareness enhancer
state_aware_input = torch.cat([state_features, states], dim=-1)
enhanced_features = self.state_aware_enhancer(state_aware_input)
```

### 3. Multi-scale Prediction
```python
# 3 parallel prediction heads
for head in self.prediction_heads:
    pred = head(temporal_features)
    multi_scale_predictions.append(pred)
```

## Data Format

Training data uses JSON format, containing:
- `instruction`: Action instruction ("wave" or "welcome")
- `observations`: Joint position observation sequences
- `joint_pos`: 26 joint position data (complete humanoid robot)

### Joint Distribution Explanation
```python
joint_indices = {
    'Leg Joints': list(range(0, 12)),    # 0-11: 12 leg joints
    'Left Arm Joints': list(range(12, 19)),   # 12-18: 7 left arm joints
    'Right Arm Joints': list(range(19, 26)),   # 19-25: 7 right arm joints
}
```

### Key Joints
- **l_arm_pitch** (index 12): Left arm pitch
- **l_arm_roll** (index 13): Left arm roll  
- **l_arm_yaw** (index 14): Left arm yaw

These joints play key roles in waving and fist-clenching actions.

Example:
```json
{
  "instruction": "wave",
  "observations": [
    {"joint_pos": [0.1, 0.2, ..., 0.3]},
    {"joint_pos": [0.1, 0.2, ..., 0.3]},
    ...
  ]
}
```

## Instruction Mapping

```python
instruction_map = {
    'wave': 0,
    'welcome': 1
}
```

## Model Files

- **Improved Model**: `checkpoints/improved_act_model_v1/best_model.pth`
- **Original Model**: `checkpoints/key_joint_act_model_v1/best_model.pth`

## Dependencies

- Python 3.8+
- PyTorch 1.12+
- ROS (Robot Operating System)
- CUDA (optional, for GPU acceleration)

## Training Tips

1. **Automatic Mixed Precision**: Enable GPU acceleration training
2. **Gradient Clipping**: Prevent gradient explosion, limit to 5.0
3. **Learning Rate Scheduling**: Use ReduceLROnPlateau
4. **Early Stopping**: Prevent overfitting, patience value 150 epochs
5. **Quality Monitoring**: Real-time training quality monitoring

## Common Issues

### Q: Dimension errors during training
A: Ensure input data dimensions are correct, state should be [batch_size, 26], action should be [batch_size, 32, 26]

### Q: Model prediction amplitude too small
A: The improved model includes action intensity regulator, should solve this problem

### Q: Trajectory connection not smooth
A: Multi-scale prediction and temporal encoding should improve connection smoothness

## Development Plan

- [ ] Add more action instructions
- [ ] Integrate visual input
- [ ] Optimize inference speed
- [ ] Add online learning functionality
- [ ] Support real-time adjustment

## Contribution Guide

1. Fork this project
2. Create feature branch
3. Submit code changes
4. Initiate Pull Request

## License

This project uses the MIT license.

## Contact

For questions or suggestions, please submit an Issue or contact the development team.