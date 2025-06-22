from pygame.time import Clock
from articulate.utils.pygame import StreamingDataViewer
from articulate.utils.wearable import WearableSensorSet
import numpy as np
import articulate as art
import matplotlib.pyplot as plt
import torch
import os
import time

class KalmanFilter:
    def __init__(self, k, b):
        
        # 初始状态估计
        x0 = np.array([[0],     # 初始高度
                       [0]])    # 初始速度
        P0 = np.eye(2) * 500    # 初始状态协方差矩阵
        self.x = x0
        self.P = P0
        
        # 定义Kalman滤波器的参数
        self.dt = 1.0 / 30.0    # 采样间隔
        self.k = k.cpu().numpy()        # pressure to height slope
        self.b = b.cpu().numpy()      # pressure to height bias
        
        # 定义状态转移矩阵
        self.F = np.array([     # 状态转移矩阵
            [1, self.dt], 
            [0, 1]
        ])
        
        # 定义控制矩阵
        self.B = np.array([     # 控制矩阵
            [0.5 * self.dt ** 2],
            [self.dt]
        ])
        
        # 定义观测矩阵
        self.H = np.array([     # 观测矩阵
            [1 / self.k, 0]
        ])
        
        # 定义过程噪声协方差矩阵
        self.Q = np.array([     # 过程噪声协方差矩阵
            [1.0, 0],
            [0, 1.0]
        ])
        
        # 定义观测噪声协方差矩阵
        self.R = np.array([[1e-3]])  # 观测噪声协方差矩阵
        
        self.I = np.eye(self.F.shape[0])  

    def predict(self, a):
        a = np.array([a]).reshape(1, 1)
        self.x = self.F @ self.x + self.B @ a
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        y = (z + self.b / self.k) - self.H @ self.x 
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
    
    def get_height(self):
        return self.x[0]

    def get_velocity(self):
        return self.x[1]

class AccIntegrator:
    def __init__(self):
        x0 = np.array([[0],     # 初始高度
                       [0]])    # 初始速度
        
        self.x = x0
        self.dt = 1.0 / 30.0
        
        # 定义状态转移矩阵
        self.F = np.array([     # 状态转移矩阵
            [1, self.dt], 
            [0, 1]
        ])
        # 定义控制矩阵
        self.B = np.array([     # 控制矩阵
            [0.5 * self.dt ** 2],
            [self.dt]
        ])

    def predict(self, a):
        # convert float to np.array
        a = np.array([a]).reshape(1, 1)
        self.x = self.F @ self.x + self.B @ a

    def get_height(self):
        return self.x[0]
    
    def get_velocity(self):
        return self.x[1]

def get_bias(k, sensor_set, b_window = 100):
    bs = []
    while True:
        data = sensor_set.get()
        if 0 in data.keys():
            pressure = data[0].pressure
            bs.append(k * pressure)
            if len(bs) > b_window:
                b = - np.mean(bs)
                print('b:', b)
                break
    return b

def get_p_bias(sensor_set, p_window = 100):
    pressures = {"sensor0": [], "sensor1": []}
    while True:
        data = sensor_set.get()
        if 0 in data.keys() and 1 in data.keys():
            pressures["sensor0"].append(data[0].pressure)
            pressures["sensor1"].append(data[1].pressure)
            if len(pressures["sensor0"]) > p_window:
                p_bias = np.mean(pressures["sensor0"]) - np.mean(pressures["sensor1"])
                print('p_bias:', p_bias)
                break
    return p_bias

def get_k_bias(sensor_set, k_window = 100, p_bias=0., delta_h = -0.6561):
    pressures = {"sensor0": [], "sensor1": []}
    while True:
        data = sensor_set.get()
        if 0 in data.keys() and 1 in data.keys():
            pressures["sensor0"].append(data[0].pressure)
            pressures["sensor1"].append(data[1].pressure + p_bias)
            if len(pressures["sensor0"]) > k_window:
                delta_p = np.mean(pressures["sensor0"]) - np.mean(pressures["sensor1"])
                k = delta_h / delta_p
                break
    return k

def test_wearable_pressure(n_calibration=2):
    clock = Clock()
    sviewer = StreamingDataViewer(1, y_range=(-1, 1), window_length=1000, names=['raw', 'filtered']); sviewer.connect()
    sensor_set = WearableSensorSet()
    
    k = - 800
    
    b_window = 100
    if n_calibration == 2:
        p_bias = get_p_bias(sensor_set)
    h_bias = get_bias(k, sensor_set, b_window)
    
    kfs = [KalmanFilter(k, h_bias) for _ in range(n_calibration)]
    while True:
        clock.tick(30)
        data = sensor_set.get()
        hs, vs, accs = [], [], []
        for i in range(n_calibration):
            if 0 in data.keys() or 1 in data.keys():
                pressure = data[i].pressure if i == 0 else data[i].pressure + p_bias
                acc = torch.tensor(data[0].raw_acceleration).float()
                ori = torch.tensor(data[0].orientation).float()
                
                a = process_acc(acc, ori)[2] * 100
                
                a = np.array([a])

                z = np.array([[pressure]])
                kfs[i].predict(a)
                kfs[i].update(z)
                h_filtered = kfs[i].get_height() * 0.01
                v_filtered = kfs[i].get_velocity() * 0.01
                
                accs.append(a / 100)
                hs.append(h_filtered)
                vs.append(v_filtered)
        
        if len(hs) != 0:
            sviewer.plot(hs)
        print('\r', clock.get_fps(), end='')

def process_acc(acc, ori):
    R = art.math.quaternion_to_rotation_matrix(ori)
    acc = R.squeeze(0).mm( - acc.unsqueeze(-1)).squeeze(-1) + torch.tensor([0, 0, - 9.8])
    return acc 

def test_wearable_acceleration():
    clock = Clock()
    sviewer = StreamingDataViewer(1, y_range=(-1, 1), window_length=200, names=['h']); sviewer.connect()
    senor_set = WearableSensorSet()
    accInt = AccIntegrator()
    
    vel = 0.0
    height = 0.0
    
    while True:
        clock.tick(30.0)
        data = senor_set.get()
        if 0 in data.keys():
            acc = torch.tensor(data[0].raw_acceleration).float() # [3,]
            ori = torch.tensor(data[0].orientation).float() # [4,]
            R = art.math.quaternion_to_rotation_matrix(ori)
            
            acc = R.squeeze(0).mm( - acc.unsqueeze(-1)).squeeze(-1) + torch.tensor([0, 0, - 9.81])
            acc_z = acc[2].item()
            
            accInt.predict(acc_z)
            h = accInt.get_height()
            v = accInt.get_velocity()
            
            delta_t = 1.0 / 30.0
            height = height + vel * delta_t + 0.5 * acc_z * delta_t ** 2
            vel = vel + acc_z * delta_t
            
            sviewer.plot([h])
            
            print('\r', clock.get_fps(), end='')

def test_raw_pressure(n_sensor = 1):
    clock = Clock()

    names = ['sensor_' + str(i) for i in range(n_sensor)]
    
    sviewer = StreamingDataViewer(n_sensor, y_range=(1003.5, 1004.5), window_length=500, names=names); sviewer.connect()
    sensor_set = WearableSensorSet()
    
    if n_sensor == 2:
        p_bias = get_p_bias(sensor_set)
    
    while True:
        clock.tick(30)
        data = sensor_set.get()
        pressures = []
        for idx in range(n_sensor):
            if idx in data.keys():
                pressure = data[idx].pressure
                if n_sensor == 2 and idx == 1:
                    pressure += p_bias
                pressures.append(pressure)
                
        if len(pressures) == n_sensor:
            sviewer.plot(pressures)

        print('\r', clock.get_fps(), end='')

def vis_dataset_heights(heights, output_dir="output_plots"):
    """
    Visualizes the relative height difference between two sensors over time and saves the plots.
    
    :param heights: List of torch.Tensors, where each tensor has shape (N, 2)
    :param output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    for idx, height in enumerate(heights):
        height = height.view(-1, 2).cpu().numpy()  # Convert to numpy if it's a tensor
        rel_height = height[:, 0] - height[:, 1]
        
        plt.figure(figsize=(24, 15))
        plt.plot(np.arange(len(rel_height)), rel_height, marker='o', linestyle='-')
        plt.xlabel("Time Step")
        plt.ylabel("Height Difference")
        plt.title(f"Relative Height Difference Over Time (Sample {idx})")
        plt.grid(True)

        # Save the plot
        plot_path = os.path.join(output_dir, f"height_diff_{idx}.png")
        plt.savefig(plot_path)
        plt.close()

    print(f"Plots saved in {output_dir}")

def livedemo(n_sensor=1):
    clock = Clock()
    delta_h = -0.6561
    names = ['sensor_' + str(i) for i in range(n_sensor)]
    
    sviewer = StreamingDataViewer(1, y_range=(-1, 1), window_length=500, names=names); sviewer.connect()
    sensor_set = WearableSensorSet()
    
    input('Stand straight in A-pose and press enter. The p_bias calibration will begin in 3 seconds')
    time.sleep(3)
    
    if n_sensor == 2:
        p_bias = get_p_bias(sensor_set)

    input('Stand straight in T-pose and press enter. The calibration will begin in 3 seconds')
    time.sleep(3)
    if n_sensor == 2:
        k = get_k_bias(sensor_set, p_bias=p_bias, delta_h=delta_h)
        print('k:', k)
    
    while True:
        clock.tick(30)
        data = sensor_set.get()
        pressures = []
        for idx in range(n_sensor):
            if idx in data.keys():
                pressure = data[idx].pressure
                if n_sensor == 2 and idx == 1:
                    pressure += p_bias
                pressures.append(pressure)
                
        if len(pressures) == n_sensor:
            h = (pressures[0] - pressures[1]) * k
            sviewer.plot([h])

        print('\r', clock.get_fps(), end='')

if __name__ == '__main__':
    # data_dir = 'data/dataset/'
    # data_name = 'totalcapture'
    
    # data_path = data_dir + data_name + '.pt'
    # data = torch.load(data_path)
   
    # output_dir = 'data/rel_heights/' + data_name + '/' 
    # os.makedirs(output_dir, exist_ok=True)
    
    # heights = data['heights']
    
    # vis_dataset_heights(heights, output_dir)
    # test_wearable_acceleration()
    # test_wearable_pressure(1)
    livedemo(2)
    # test_raw_pressure(1)
    