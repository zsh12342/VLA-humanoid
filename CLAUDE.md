# VLA Project Development Rules and Constraints

## Project Overview
This is a Visual-Language Action (VLA) prediction project that predicts robot arm movements. The current implementation uses a fixed dataset with 8 repetitions of the same action, collected at high frame rates resulting in small inter-frame differences that are difficult to predict.

## Core Problem Statement
- **Dataset Issue**: Fixed action collected 8 times with high frame rate
- **Prediction Challenge**: Small inter-frame differences make prediction difficult
- **Current Approach**: Predict change amounts but output target positions
- **Key Problem**: Good dataset fitting but poor real-world inference performance

## Development Rules and Constraints

### 1. Code Modification Rules
- **Single File Modification**: ALL training model code changes MUST be made in `/root/kuavo_ws/src/vla/simple_act_train.py` only
- **No New Files**: Do NOT create new files for training model modifications
- **Error Prevention**: Each code modification must avoid previously encountered problems
- **Deep Thinking**: Apply deep thinking to ensure every modification is effective and necessary

### 1.5 CRITICAL PRINCIPLE: TWO-LAYER MODEL ARCHITECTURE
- **Fundamental Architecture**: Model MUST use two-layer structure: Instruction Classification + Action Prediction
- **Instruction Uniqueness**: Each instruction must map to exactly one unique action pattern
- **No Action Overlap**: Wave and welcome instructions must produce completely different actions
- **Classification Layer**: First layer classifies instructions with cross-entropy loss (weight: 10.0)
- **Independent Predictors**: Second layer uses independent networks for each instruction's action prediction
- **Strict Diversity Loss**: Must enforce large differences between instruction patterns (weight: 5.0)
- **Success Metric**: Instruction classification accuracy > 80% AND pattern difference > 1.0

### 2. Model Evaluation Rules
- **Dataset Fitting Validation**: Model success MUST be validated by running `/root/kuavo_ws/src/vla/model_vs_robot_comparison.py` to compare model output with dataset fitting
- **Generalization Gap Solution**: MUST solve the common problem where dataset fits perfectly but inference performs poorly
- **Training Speed Detection**: If training converges extremely fast with loss/accuracy stabilizing in first few epochs, this indicates model design error and inability to learn proper patterns

### 3. Model Performance Requirements
- **Generalization**: The model must be general-purpose, not over-constrained for specific examples
- **Direction Accuracy**: Direction prediction must be 100% accurate - failure indicates complete training strategy failure
- **No Low-Level Errors**: Avoid basic mistakes in implementation
- **Real-World Performance**: Must solve the gap between good dataset fitting and poor inference performance
- **Dataset Fitting First**: Model must first demonstrate accurate dataset fitting before any inference claims

### 4. Technical Constraints
- **Time Information**: MUST add time information as auxiliary input during model training to address linear joint growth issues
- **Change Prediction**: Focus on predicting change amounts while maintaining target position output
- **Signal Enhancement**: Address weak signal issues from high frame rate data collection
- **Non-Linear Movement**: Prevent single joint linear growth patterns in inference
- **IMPORTANT**: The most critical issue is NOT direction accuracy - the model's evaluation should focus solely on whether the change amount prediction is accurate. Direction accuracy is not the evaluation standard; only the accuracy of change amount prediction matters.

### 5. Training Strategy Requirements
- **Cumulative Learning**: Consider cumulative changes rather than instantaneous changes for stronger learning signals
- **Scale Factor Learning**: Learn appropriate scaling factors for change predictions
- **Critical Joint Focus**: Pay special attention to critical joints (l_arm_pitch, l_arm_roll, l_arm_yaw - indices 12, 13, 14)
- **Loss Function Design**: Use comprehensive loss functions that ensure direction accuracy and magnitude matching
- **Progressive Learning**: Training should show gradual improvement, not immediate convergence

### 6. Quality Assurance
- **No Shortcut Solutions**: Avoid solutions that sacrifice generalization for specific case performance
- **Direction Priority**: Direction accuracy is paramount - if direction is wrong, the entire approach fails
- **Iterative Improvement**: Each modification should build upon previous learnings
- **Validation Focus**: Ensure improvements work in real inference, not just on training data

## Current Implementation Notes

### Model Architecture - TWO-LAYER STRUCTURE (CRITICAL PRINCIPLE)
- Uses `SimpleACTGenerator` class - **Two-layer model structure: Instruction Classification + Action Prediction**
- **Layer 1: Instruction Classifier** - Ensures instruction uniqueness with cross-entropy loss
- **Layer 2: Instruction-Specific Action Predictors** - Independent networks for each instruction
- **Principle: Each instruction maps to exactly one unique action pattern, no overlap allowed**
- VAE latent space for learning diverse action patterns
- LSTM decoder for temporal sequence generation
- Multi-component loss function addressing small variable problems
- Strict instruction diversity loss ensuring different instructions produce completely different actions

### Training Configuration
- **Latest Training script**: `enhanced_smooth_act_train.py`
- **Latest Inference script**: `enhanced_smooth_act_inference_node.py`
- Batch size: 32
- Learning rate: 5e-4 with CosineAnnealing scheduling
- Hidden dimension: 256
- Action dimension: 26
- State dimension: 26
- Trajectory length: 100
- Training epochs: 2000 (with early stopping)

### Enhanced Model Architecture
- **Multi-head Action Prediction**: 3 parallel prediction heads for stability
- **Small Variable Enhancement Head**: Specialized head for learning small changes
- **Transformer Decoder**: Better temporal modeling than LSTM
- **Enhanced Instruction Encoding**: Deeper instruction embedding network
- **Temporal Encoding**: Learnable temporal patterns for better sequence modeling

### Advanced Loss Components
- **Small Variable Loss**: Enhanced learning for small joint movements
- **Change Consistency Loss**: Ensures predicted changes are physically plausible
- **Temporal Smoothness Loss**: Guarantees smooth transitions between timesteps
- **Mid-Activity Loss**: Prevents static behavior in middle trajectory segments
- **Diversity Loss**: Ensures different instructions produce different trajectories
- **Multi-Objective Optimization**: Balanced combination of all loss components

### Inference Requirements
- **Critical**: Must use correct ROS topics: `/humanoid_controller/optimizedState_mrt/joint_pos`
- **Critical**: Must use `Float64MultiArray` message type, not `JointState`
- **Critical**: Must use Chinese instruction mapping: `{'挥手': 0, '抱拳': 1}`
- **Critical**: Must publish arm commands using `armTargetPoses` message format
- **Critical**: Must set arm control mode to EXTERN_CONTROL (mode 2)

### Trajectory Execution Rules - CONTINUOUS EXECUTION REQUIREMENT
1. **Continuous Segment Generation**: Generate trajectory segments (100 steps each) continuously to form long action sequences
2. **Seamless Transition**: Generate next segment 20 steps before current segment completes, ensuring no gaps between segments
3. **Predictive Handover**: Use predicted final state of current segment as starting state for next segment generation
4. **Dynamic Trajectory Extension**: Concatenate trajectory segments in real-time to form continuous long trajectories
5. **No Interruption**: Maintain constant 10Hz execution frequency without any pauses or waiting between segments
6. **Uninterrupted Flow**: Create the illusion of infinite continuous motion by seamlessly chaining trajectory segments

### Enhanced Inference Implementation Details:
- **Advanced Lookahead**: 25 steps提前生成下一段轨迹（更早准备）
- **Gaussian Smoothing**: Use scipy.ndimage.gaussian_filter1d for trajectory smoothing
- **Smooth Blending**: 10-step blending between trajectory segments for seamless transition
- **Higher Frequency**: 20Hz execution rate for smoother motion
- **Multi-Head Prediction**: Use ensemble of prediction heads for more stable output
- **Adaptive State Management**: Intelligent trajectory extension with smooth handover

### Inference Flow Improvements:
1. **Trajectory Smoothing**: Apply Gaussian filtering to all generated segments
2. **Predictive Blending**: Smooth transition between old and new trajectory segments
3. **Frequency Enhancement**: Double execution frequency (20Hz) for smoother appearance
4. **Ensemble Prediction**: Use multiple prediction heads and average/blend results
5. **Real-time Adaptation**: Continuously adapt to current robot state during execution

### Performance Requirements
- **Direction Accuracy**: 100% correct direction prediction
- **Instruction Differentiation**: Different instructions must produce visibly different trajectories
- **No Stuck Behavior**: Model must not get stuck outputting nearly constant values
- **Temporal Consistency**: Predictions must show proper temporal evolution
- **Real-World Performance**: Must work on actual robot, not just training data

### Critical Issues to Address - CURRENT PROBLEMS
1. **Small Variable Problem in Mid-Trajectory**: 
   - Model gets stuck in middle segments where joint changes are very small
   - Need enhanced signal amplification for small movements
   - Must maintain dynamic behavior throughout entire trajectory, not just beginning

2. **Inference Smoothness Issue**:
   - Current execution still shows visible discontinuities between segments
   - Need better trajectory blending and transition smoothing
   - Must achieve truly fluid motion that appears as single continuous action

3. **Progressive Motion Requirement**:
   - Model must generate meaningful changes at every timestep, not just initial segments
   - Avoid "static middle" problem where motion stops in middle of trajectory
   - Ensure consistent motion dynamics throughout entire execution

### Training Focus Areas for Next Iteration
- **Enhanced Small Variable Learning**: Strengthen model's ability to learn and predict small joint movements
- **Improved Temporal Consistency**: Ensure smooth transitions and consistent motion throughout trajectory
- **Better Signal-to-Noise Ratio**: Amplify small signals while maintaining training-inference consistency
- **Dynamic Middle Segments**: Focus on eliminating static behavior in middle portions of trajectories

### Key Improvements
- **Solved inference getting stuck problem**: Model now predicts dynamic changes
- **Enhanced temporal modeling**: Better position encoding and attention mechanisms
- **Multi-objective loss**: Action prediction + change prediction + consistency constraints
- **Proper normalization**: ACT-style dataset statistics normalization

### Current Performance
- MSE: 0.029 (acceptable, shows dynamic behavior)
- Average correlation: 0.622 (significant improvement from 0.0955)
- Change ratio: 1.405 (predictions match target dynamics)
- Critical joints (12,13,14) show good correlations: 0.862, 0.941, 0.944

### Loss Function Components
1. **Basic MSE Loss**: Primary loss component
2. **Cumulative Change Loss**: Stronger learning signal
3. **Critical Joint Weighted Loss**: Focus on important joints
4. **Magnitude Matching**: Constrains prediction scales
5. **Joint-wise Magnitude Loss**: Per-joint scale constraints

## Development Guidelines

### When Modifying Code:
1. **Analyze First**: Understand the current implementation thoroughly
2. **Plan Changes**: Consider impact on all system components
3. **Test Incrementally**: Verify each change works as expected
4. **Document Changes**: Keep track of modifications and their effects

### 7. Critical Validation Rules - NEVER VIOLATE
- **Real Validation Requirement**: Model validation MUST use `model_vs_robot_comparison.py` to test actual fitting performance
- **No False Convergence Claims**: NEVER claim a model is "reasonable" or "learning properly" based solely on training metrics
- **Magnitude Accuracy Requirement**: Model output magnitude must be within 2x of dataset magnitude (currently 11.6x difference is unacceptable)
- **Training-Validation Consistency**: Training validation metrics MUST correlate with actual model performance
- **No Premature Success Claims**: NEVER claim success until model_vs_robot_comparison.py shows MSE < 0.1 and direction accuracy > 90%
- **Honest Assessment**: Always run actual validation before making any claims about model performance

## Common Pitfalls to Avoid:
- Overfitting to training data while ignoring inference performance
- Neglecting time information in model architecture
- Creating solutions that work for specific cases but lack generalization
- Ignoring direction accuracy in favor of magnitude precision
- **CRITICAL**: Claiming model success based on training metrics alone without validation
- **CRITICAL**: Accepting false convergence where validation metrics look good but actual performance is poor
- **CRITICAL**: Ignoring large magnitude differences (>2x) between model output and dataset values

### Success Metrics:
- **Direction Accuracy**: 100% correct direction prediction
- **Inference Performance**: Good real-world performance matching training results
- **Generalization**: Works across different action types and conditions
- **Non-Linear Movements**: Avoids single joint linear growth patterns

## Remember
- This is a general model, not a specific case solver
- Direction accuracy is non-negotiable
- Time information is crucial for realistic movement prediction
- Every modification must be thoughtful and effective

### Language Rule
- **Chinese Communication**: All communication, documentation, and comments must be in Chinese (中文)