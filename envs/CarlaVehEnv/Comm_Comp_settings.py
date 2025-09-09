import os
from envs.CarlaVehEnv.compute_K_star import find_K_star_mc
# from compute_K_star import find_K_star, find_K_star_mc
import numpy as np
import matplotlib.pyplot as plt
import logging
LOG = logging.getLogger(__name__)

class Comm_Comp_Base():
    def __init__(self, configs):
        self.type = 'Base'

        self.Bandwidth = 50# configs.Bandwidth
        self.Up_ratio = 0.6
        self.Down_ratio = 0.3
        self.guard_ratio = 0.1
        self.protocol_effi = 0.8
        # data size unit MB
        self.data_size_summary = 0.005
        self.data_size_map = 2
        # process time unit s 
        self.proc_time_per_map = 0.01
        self.spectral_efficiency = 4.5
        self.Frame_T = 0.2

    def get_uplink_rate(self):
        """
        formula: R_UL = B * eta * protocol_effi * Up_ratio
        B: MHz、eta: bits/s/Hz
        """
        B_Hz = self.Bandwidth * 1e6  
        R_bits_per_s = B_Hz * self.spectral_efficiency * self.protocol_effi * self.Up_ratio
        return R_bits_per_s / 1e6                     

    def get_downlink_rate(self):
        """
        return Mbps
        formula: R_UL = B * eta * protocol_effi * Down_ratio
        B: MHz、eta: bits/s/Hz
        """
        B_Hz = self.Bandwidth * 1e6
        R_bits_per_s = B_Hz * self.spectral_efficiency * self.protocol_effi * self.Down_ratio
        return R_bits_per_s / 1e6
    
    def get_time_up0(self):
        """
        return: s
        """
        up_link_rate = self.get_uplink_rate()
        t = self.data_size_summary / up_link_rate
        return t

    def get_time_up1(self, map_size, area_count):
        """
        return: s
        """
        up_link_rate = self.get_uplink_rate()
        t = (self.data_size_map / map_size) * area_count / up_link_rate
        return t

    def get_time_down(self):
        """
        return: s
        """
        down_link_rate = self.get_downlink_rate()
        t = self.data_size_summary / down_link_rate
        return t

    def get_time_proc(self, map_size, area_cnt):
        """
        return: s
        """

        return self.proc_time_per_map / map_size * area_cnt    

    def _print(self, map_size, area_cnt):
        up = self.get_uplink_rate()
        down = self.get_downlink_rate()
        up_time0 = self.get_time_up0()
        up_time1 = self.get_time_up1(map_size, area_cnt)
        down_time = self.get_time_down()
        proc_time = self.get_time_proc(map_size, area_cnt)
        LOG.info(f"Current Communication & Computing model settings: \n \
                 \t map_size: {map_size}, area_cnt {area_cnt} \n \
                 \t type: {self.type} \n \
                 \t Bandwidth: {self.Bandwidth} MHz \n \
                 \t Uplink rate: {up} MB, Downlink rate {down} MB \n \
                 \t up_time0: {up_time0}, up_time1 {up_time1} \n \
                 \t down_time: {down_time}, proc_time {proc_time}")
        
    def get_latency(self, map_size, area_cnt, beta = 0.001):
        times_delay_cop = self.get_time_up1(map_size, area_cnt)  + self.get_time_proc(map_size, area_cnt) # + self.get_time_up0() + self.get_time_down() 
        decay = np.exp(-times_delay_cop)
        return decay * beta
        
    
    def compute_k_star(self, agent_num, map_size, lamda):
        s0 = self.data_size_summary / map_size
        s1 = self.data_size_map / map_size
        Rul = self.get_uplink_rate()     
        Rdl = self.get_downlink_rate()  
         

        K_star, info  = find_K_star_mc(
            N=agent_num, U=map_size, cover_prob=lamda,
            s0_MB=s0, s1_MB=s1,
            Rul_Mbps=Rul, Rdl_Mbps=Rdl,
            T_s=self.Frame_T, K_max_cap=None, beta=None#,  # beta=None => 工程速率版时延
            #latecy_fn=self.get_latency
        )

        # print(f"info: {info}")
        # table, K_budget, K_max = info['table'], info['Kmax_budget'], info['Kmax_used']
        # print(f"K_budget (frame UL capacity upper bound) = {K_budget}")
        # print(f"Search K in [0, {K_max}] -> K* = {K_star}")
        return K_star, info

class Lower_Resource(Comm_Comp_Base):
    def __init__(self, configs):
        super().__init__(configs)
        self.type = "Lower"
        self.Up_ratio = 0.6
        self.Down_ratio = 0.3
        self.guard_ratio = 0.1
        self.protocol_effi = 0.8
        # data size unit MB
        self.data_size_summary = 0.005
        self.data_size_map = 2
        # process time unit s 
        self.proc_time_per_map = 0.01
        self.spectral_efficiency = 4.5
    
if __name__ == "__main__":
    N = 5            # 车辆数
    U = 1600          # 候选区域数（同一帧）
    lamba = 0.30     # 单车覆盖该区域的伯努利参数（可由你的 IoU/λ 推得）

    comm = Comm_Comp_Base(None)  # 20 MHz, 200ms 帧
    K_star, info  = comm.compute_k_star(N, U, lamba)
    print(f"info: {info}")
    table, K_budget, K_max = info['table'], info['Kmax_budget'], info['Kmax_used']
    print(f"K_budget (frame UL capacity upper bound) = {K_budget}")
    print(f"Search K in [0, {K_max}] -> K* = {K_star}")

    # 画 J(K) 与 ΔJ(K)
    # table = info["table"]                     # 你打印出来的那个 dict
    Ks = sorted(table.keys())
    vals = [float(table[k]["value"]) for k in Ks]

    # 现算 ΔJ(K) = J(K) - J(K-1)，并保证长度一致
    deltas = [vals[0]] + [vals[i] - vals[i-1] for i in range(1, len(vals))]

    #（可选）也取出 ERK 和 discount 画/打印看看
    ERKs  = [float(table[k]["ERK"])  for k in Ks]
    discs = [float(table[k]["disc"]) for k in Ks]

    # 画图
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,3))
    plt.plot(Ks, vals, marker='o')
    plt.title("J(K)")
    plt.xlabel("K"); plt.ylabel("J(K)")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(6,3))
    plt.plot(Ks, deltas, marker='o')
    plt.axhline(0.0, linestyle='--')
    plt.title("ΔJ(K) = J(K) - J(K-1)")
    plt.xlabel("K"); plt.ylabel("ΔJ(K)")
    plt.tight_layout(); plt.show()