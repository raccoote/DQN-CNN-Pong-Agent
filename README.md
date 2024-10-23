# DQN-CNN-Pong-Agent

This model uses the Gymnasium library to create the Atari game environment and PyTorch for building and training the neural network. The left paddle is following the y position of the puck, while the right one is the implementaion of our DQN agent.

![result](https://github.com/user-attachments/assets/7ebc4c2f-aa6e-4924-add6-2afab10b9056)

In this PONGAtari implementation, RGB images are used as input, each with 3 color channels, a height of 210 pixels, and a width of 160 pixels.

The first convolutional layer has 32 filters (8x8 kernel, stride 4, padding 2) followed by ReLU. The second layer has 64 filters (4x4 kernel, stride 2, padding 1), also followed by ReLU. The third layer uses 64 filters (3x3 kernel, stride 1, padding 1) with ReLU.

The output of the convolutional layers is flattened using x.reshape(x.size(0), -1) before passing to fully connected layers, which predict the agent's actions based on the extracted features. Pong has 3 discrete actions: UP, DOWN, and NOOP (no operation). The first fully connected layer has 512 neurons with ReLU, and the second layer outputs the final action predictions.



![image](https://github.com/user-attachments/assets/127f83cf-deb9-44c8-a8f0-c096d5a85ccd)




# Dependencies
```
>> pip install torch
>> pip install gymnasium [ classic_control ]
>> pip install gymnasium [ atari ]
>> pip install gymnasium [ accept - rom - license ]
```



# Hyperparameters



First Run

• BATCH SIZE = 8: Chosen small due to limited computational resources.


• GAMMA = 0.99: Discount factor for future rewards.

• EPS START = 0.9, EPS END = 0.05, EPS DECAY = 1000: Maintains exploration during training.

• TAU = 0.005: Target network update coefficient.

• LR = 0.0001: Learning rate.

• REPLAY MEMORY SIZE = 100,000: Access to a wide range of experiences.

• RENDER = False: Whether to visualize the game.

• NUM EPISODES = 50:


![image](https://github.com/user-attachments/assets/1419dff0-4d5c-4029-8fcd-907f79495beb)


Second Run

• BATCH SIZE = 64

• REPLAY MEMORY SIZE = 100.000

• TAU = 0.001

• GAMMA = 0.95




![image](https://github.com/user-attachments/assets/858c4e1c-e81c-4b99-8d56-61a7c2972316)


