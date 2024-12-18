import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import InterpolatedUnivariateSpline
import datetime
import logging
from collections import deque
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesPredictor:
    """Neural network for time series prediction of sensor data"""
    def __init__(self, sequence_length: int = 50, forecast_horizon: int = 10):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler()
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            layers.LSTM(64, input_shape=(self.sequence_length, 1), return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(self.forecast_horizon)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model

    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.sequence_length - self.forecast_horizon):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[(i + self.sequence_length):(i + self.sequence_length + self.forecast_horizon)])
        return np.array(X), np.array(y)

    def train(self, data: np.ndarray, epochs: int = 50, batch_size: int = 32) -> Dict:
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = self.prepare_sequences(scaled_data)
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        return history.history

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        scaled_sequence = self.scaler.transform(sequence.reshape(-1, 1))
        scaled_prediction = self.model.predict(
            scaled_sequence.reshape(1, self.sequence_length, 1),
            verbose=0
        )
        return self.scaler.inverse_transform(scaled_prediction)

class DRLAgent:
    """Deep Reinforcement Learning agent for rocket control"""
    def __init__(self, state_size: int, action_size: int, memory_size: int = 100000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            layers.Dense(128, input_dim=self.state_size, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size: int = 32) -> float:
        if len(self.memory) < batch_size:
            return 0.0

        minibatch = random.sample(self.memory, batch_size)
        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (
            np.amax(self.target_model.predict(next_states, verbose=0), axis=1)
        ) * (1 - dones)
        
        target_f = self.model.predict(states, verbose=0)
        
        for i, action in enumerate(actions):
            target_f[i][action] = targets[i]

        history = self.model.fit(states, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return history.history['loss'][0]

class EnhancedRocketSimulationAgent:
    """Enhanced rocket simulation agent with DRL capabilities"""
    def __init__(self, mass: float, initial_position: np.ndarray, initial_velocity: np.ndarray):
        self.mass = mass
        self.position = initial_position
        self.velocity = initial_velocity
        self.acceleration = np.zeros(3)
        self.orientation = np.array([0.0, 0.0, 0.0])
        self.gimbal_angle = np.array([0.0, 0.0, 0.0])
        self.thrust_vector = np.array([0.0, 0.0, 0.0])
        
        # Initialize DRL agent
        self.state_size = 18  # position(3), velocity(3), acceleration(3), orientation(3), gimbal(3), thrust(3)
        self.action_size = 6  # gimbal adjustments(3) and thrust adjustments(3)
        self.drl_agent = DRLAgent(self.state_size, self.action_size)
        
        # PID controller gains
        self.Kp = 1.0
        self.Ki = 0.1
        self.Kd = 0.05
        self.integral_error = np.zeros(3)
        self.last_error = np.zeros(3)
        
        # Performance tracking
        self.performance_history = {
            'position_error': [],
            'velocity_error': [],
            'rewards': [],
            'thrust_adjustments': [],
            'gimbal_adjustments': []
        }

    def get_state(self) -> np.ndarray:
        """Get current state vector for DRL"""
        return np.concatenate([
            self.position,
            self.velocity,
            self.acceleration,
            self.orientation,
            self.gimbal_angle,
            self.thrust_vector
        ])

    def update_sensor_data(self, accelerometer_data: np.ndarray, gyroscope_data: np.ndarray,
                          gps_data: np.ndarray, pressure_data: float, temp_data: float,
                          density_data: float) -> None:
        """Update sensor data and calculate reward for DRL"""
        previous_state = self.get_state()
        
        # Update sensor readings
        self.acceleration = accelerometer_data
        self.orientation = gyroscope_data
        self.position = gps_data
        self.pressure = pressure_data
        self.temperature = temp_data
        self.density = density_data
        
        # Calculate reward based on trajectory improvement
        current_error = self.calculate_trajectory_error(self.position, self.velocity)
        previous_error = self.calculate_trajectory_error(
            previous_state[:3],  # Previous position
            previous_state[3:6]  # Previous velocity
        )
        
        reward = previous_error - current_error  # Positive if error decreased
        
        # Store experience in DRL agent
        current_state = self.get_state()
        self.drl_agent.remember(
            previous_state,
            0,  # Action will be determined by DRL agent
            reward,
            current_state,
            False
        )
        
        # Train DRL agent
        self.drl_agent.replay()

    def calculate_trajectory_error(self, position: np.ndarray, velocity: np.ndarray) -> float:
        target_position = np.array([0.0, 0.0, 100.0])
        target_velocity = np.array([0.0, 0.0, 5.0])
        
        position_error = np.linalg.norm(position - target_position)
        velocity_error = np.linalg.norm(velocity - target_velocity)
        
        return position_error + velocity_error

    def calculate_correction(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate corrections using both PID and DRL"""
        # Get DRL action
        state = self.get_state()
        action = self.drl_agent.act(state)
        
        # Convert action to gimbal and thrust adjustments
        drl_gimbal_adjustment = np.array([
            (action % 3 - 1) * 0.1,  # -0.1, 0, or 0.1 for each axis
            ((action // 3) % 3 - 1) * 0.1,
            ((action // 9) % 3 - 1) * 0.1
        ])
        
        # Calculate PID control
        error = self.calculate_trajectory_error(self.position, self.velocity)
        self.integral_error += error
        derivative_error = error - self.last_error
        
        pid_correction = (
            self.Kp * error +
            self.Ki * self.integral_error +
            self.Kd * derivative_error
        )
        
        # Combine PID and DRL corrections
        gimbal_correction = 0.7 * pid_correction + 0.3 * drl_gimbal_adjustment
        thrust_correction = self.calculate_thrust_adjustment(self.pressure, self.density)
        
        # Update performance history
        self.performance_history['position_error'].append(error)
        self.performance_history['thrust_adjustments'].append(np.linalg.norm(thrust_correction))
        self.performance_history['gimbal_adjustments'].append(np.linalg.norm(gimbal_correction))
        
        self.last_error = error
        return gimbal_correction, thrust_correction

    def calculate_thrust_adjustment(self, pressure: float, density: float) -> np.ndarray:
        """Calculate thrust adjustments based on atmospheric conditions"""
        # Base adjustment from original implementation
        if pressure < 0.8:
            thrust_adjustment = np.array([0.05, 0.0, 0.0])
        elif density < 1.0:
            thrust_adjustment = np.array([0.02, 0.0, 0.0])
        else:
            thrust_adjustment = np.array([0.0, 0.0, 0.0])
        
        # Get DRL adjustment
        state = self.get_state()
        action = self.drl_agent.act(state)
        
        # Modify thrust based on DRL suggestion
        drl_adjustment = np.array([
            ((action // 27) % 3 - 1) * 0.02,
            ((action // 81) % 3 - 1) * 0.02,
            ((action // 243) % 3 - 1) * 0.02
        ])
        
        return thrust_adjustment + drl_adjustment

    def simulate_step(self, time_step: float) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate one time step with physics calculations"""
        # Calculate corrections
        gimbal_correction, thrust_correction = self.calculate_correction()
        
        # Apply corrections
        self.gimbal_angle += gimbal_correction
        self.thrust_vector += thrust_correction
        
        # Update rocket state
        gravity = np.array([0, 0, -9.81])
        
        # Calculate forces
        thrust_force = self.thrust_vector * np.cos(np.linalg.norm(self.gimbal_angle))
        
        # Calculate acceleration
        self.acceleration = (thrust_force / self.mass) + gravity
        
        # Update position and velocity using basic physics
        self.velocity += self.acceleration * time_step
        self.position += self.velocity * time_step + 0.5 * self.acceleration * time_step**2
        
        return self.position, self.velocity

class EnhancedSensorData:
    """Enhanced sensor data handling with LSTM predictions"""
    def __init__(self, sequence_length: int = 50):
        self.sequence_length = sequence_length
        self.data_buffer = {
            'accelerometer': np.zeros((sequence_length, 3)),
            'gyroscope': np.zeros((sequence_length, 3)),
            'gps': np.zeros((sequence_length, 3)),
            'pressure': np.zeros(sequence_length),
            'temperature': np.zeros(sequence_length),
            'density': np.zeros(sequence_length),
            'wind_velocity': np.zeros((sequence_length, 3)),
            'humidity': np.zeros(sequence_length),
            'altitude': np.zeros(sequence_length)
        }
        
        # Initialize predictors
        self.predictors = {
            'pressure': TimeSeriesPredictor(sequence_length),
            'temperature': TimeSeriesPredictor(sequence_length),
            'density': TimeSeriesPredictor(sequence_length),
            'humidity': TimeSeriesPredictor(sequence_length),
            'altitude': TimeSeriesPredictor(sequence_length)
        }
        
        # Initialize vector component predictors
        for i in range(3):
            for sensor in ['accelerometer', 'gyroscope', 'gps', 'wind_velocity']:
                self.predictors[f'{sensor}_{i}'] = TimeSeriesPredictor(sequence_length)

        self.timestamp = []

    def update(self, new_data: Dict[str, np.ndarray]) -> None:
        """Update sensor readings and maintain historical buffer"""
        
        current_time = datetime.datetime.now()
        self.timestamp.append(current_time)
        
        for key, value in new_data.items():
            if key in self.data_buffer:
                if isinstance(value, (np.ndarray, list)) and len(value) == 3:
                    self.data_buffer[key] = np.roll(self.data_buffer[key], -1, axis=0)
                    self.data_buffer[key][-1] = np.array(value)
                elif isinstance(value, (int, float)):
                    self.data_buffer[key] = np.roll(self.data_buffer[key], -1)
                    self.data_buffer[key][-1] = float(value)
                else:
                    logger.warning(f"Invalid data format for sensor {key}")
            else:
                logger.warning(f"Unknown sensor type: {key}")

    def train_predictors(self, historical_data: Dict[str, np.ndarray]) -> None:
        """Train all predictors using historical data"""
        for key, predictor in self.predictors.items():
            if '_' in key:
                base_key, index = key.split('_')
                data = historical_data[base_key][:, int(index)]
            else:
                data = historical_data[key]
            
            predictor.train(data)

    def predict_future_values(self, steps: int) -> Dict[str, np.ndarray]:
        """Predict future sensor values"""
        predictions = {}
        
        # Predict scalar quantities
        for key in ['pressure', 'temperature', 'density', 'humidity', 'altitude']:
            sequence = self.data_buffer[key]
            predictions[key] = self.predictors[key].predict(sequence)[-steps:]
        
        # Predict vector quantities
        vector_quantities = ['accelerometer', 'gyroscope', 'gps', 'wind_velocity']
        for qty in vector_quantities:
            predictions[qty] = np.zeros((steps, 3))
            for i in range(3):
                sequence = self.data_buffer[qty][:, i]
                predictions[qty][:, i] = self.predictors[f'{qty}_{i}'].predict(sequence)[-steps:]
        
        return predictions

def main():
    """Main function to demonstrate the integrated system"""
    # Initialize systems
    rocket = EnhancedRocketSimulationAgent(
        mass=5000,
        initial_position=np.array([0.0, 0.0, 0.0]),
        initial_velocity=np.array([0.0, 0.0, 0.0])
    )
    sensor_system = EnhancedSensorData()
    
    # Simulation parameters
    simulation_steps = 1000
    time_step = 0.01
    prediction_horizon = 10
    
    # Performance tracking
    trajectory_errors = []
    prediction_errors = []
    
    # Generate historical data for initial training
    historical_data = {
        'accelerometer': np.random.normal(0, 1, (1000, 3)),
        'gyroscope': np.random.normal(0, 0.1, (1000, 3)),
        'gps': np.zeros((1000, 3)),
        'pressure': 101325 * np.exp(-np.arange(1000) * 10 / 7400),
        'temperature': 288.15 - 0.0065 * np.arange(1000) * 10,
        'density': 1.225 * np.exp(-np.arange(1000) * 10 / 7400),
        'wind_velocity': np.random.normal(0, 5, (1000, 3)),
        'humidity': 50 + np.random.normal(0, 5, 1000),
        'altitude': np.arange(1000) * 10
    }
    
    # Train sensor predictors
    logger.info("Training sensor predictors...")
    sensor_system.train_predictors(historical_data)
    
    # Main simulation loop
    logger.info("Starting simulation...")
    for step in range(simulation_steps):
        # Simulate sensor readings
        sensor_data = {
            'accelerometer': rocket.acceleration,
            'gyroscope': rocket.orientation,
            'gps': rocket.position,
            'pressure': 101325 * np.exp(-rocket.position[2] / 7400),
            'temperature': 288.15 - 0.0065 * rocket.position[2],
            'density': 1.225 * np.exp(-rocket.position[2] / 7400),
            'wind_velocity': np.random.normal(0, 5, 3),
            'humidity': 50 + np.random.normal(0, 5),
            'altitude': rocket.position[2]
        }
        
        # Update systems
        sensor_system.update(sensor_data)
        rocket.update_sensor_data(
            sensor_data['accelerometer'],
            sensor_data['gyroscope'],
            sensor_data['gps'],
            sensor_data['pressure'],
            sensor_data['temperature'],
            sensor_data['density']
        )
        
        # Simulate rocket movement
        new_position, new_velocity = rocket.simulate_step(time_step)
        
        # Make predictions if enough data collected
        if step > 50:
            future_predictions = sensor_system.predict_future_values(steps=prediction_horizon)
            
            # Calculate prediction errors
            actual_altitude = sensor_data['altitude']
            predicted_altitude = future_predictions['altitude'][0]
            prediction_error = np.abs(actual_altitude - predicted_altitude)
            prediction_errors.append(prediction_error)
            
            # Calculate trajectory error
            trajectory_error = rocket.calculate_trajectory_error(new_position, new_velocity)
            trajectory_errors.append(trajectory_error)
            
            # Log progress
            if step % 100 == 0:
                logger.info(f"Step {step}")
                logger.info(f"Position: {new_position}")
                logger.info(f"Velocity: {new_velocity}")
                logger.info(f"Altitude prediction error: {prediction_error:.2f}")
                logger.info(f"Trajectory error: {trajectory_error:.2f}")
                logger.info(f"DRL epsilon: {rocket.drl_agent.epsilon:.3f}")
    
    # Final performance analysis
    logger.info("\nSimulation Complete")
    logger.info(f"Average prediction error: {np.mean(prediction_errors):.2f}")
    logger.info(f"Average trajectory error: {np.mean(trajectory_errors):.2f}")
    logger.info(f"Final position: {rocket.position}")
    logger.info(f"Final velocity: {rocket.velocity}")
    
    # Save performance history
    performance_data = {
        'trajectory_errors': trajectory_errors,
        'prediction_errors': prediction_errors,
        'position_errors': rocket.performance_history['position_error'],
        'thrust_adjustments': rocket.performance_history['thrust_adjustments'],
        'gimbal_adjustments': rocket.performance_history['gimbal_adjustments']
    }
    
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv('simulation_performance.csv', index=False)
    logger.info("Performance data saved to simulation_performance.csv")

if __name__ == "__main__":
    main()
