[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atalaydenknalbant/ML-Displacement-Detection/blob/main/TF_Displacement.ipynb) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fatalaydenknalbant%2FML-Displacement-Detection&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# Video Based On Dynamic Testing For A Lab Scale Frame Structure

This repository holds code and resources for a project that applies deep neural network regression to predict frame structure durability from vibration data. The core concept involves measuring real frame structure vibrations with an ADXL335 accelerometer sensor mounted on an Arduino microcontroller. The sensor data are synchronized with recorded video reference frames to allow visual validation and post analysis.

## Data Acquisition

The frame structure is subjected to dynamic loads in a laboratory setting. The ADXL335 analog accelerometer sensor is mounted at key locations on the frame. An Arduino board samples the sensor outputs at 1000 samples per second. Each reading is timestamped and saved to a CSV file while the experiment is recorded on video for reference.

## Data Processing

Raw acceleration values undergo low pass filtering to remove high frequency noise. Each filtered trace is integrated once to obtain velocity and then again to derive displacement over time. The displacement time series are aligned with frame indices from the video to confirm physical consistency.

## Model Design

The regression model runs on TensorFlow 2 and accepts fixed-length vibration segments as input. It uses two hidden layers of 64 neurons each with relu activation and outputs a single durability metric corresponding to the maximum displacement capacity under load.

## Training Process

Training uses mean squared error loss and the Adam optimizer (learning_rate=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-05, amsgrad=False). The model trains for up to 1000 epochs with 10% validation split, employs early stopping (patience=60, min_delta=0.01), and tracks mae and mse metrics.

## Evaluation

Performance is measured on held-out vibration data using mean absolute error (MAE) and mean squared error (MSE). Final validation on a fully assembled frame under realistic dynamic loads confirms that predicted durability metrics match experimental measurements without systematic bias.

## License

This project is under the MIT license. See the [LICENSE](LICENSE) file for details.
