#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成品油二次配送车辆路径问题遗传算法解决方案（改进版）
考虑多车型、软时间窗、隔舱运输约束的成品油配送优化
支持中转路径，使用Floyd-Warshall算法计算最短路径

主要改进：
1. 使用Floyd-Warshall算法计算所有节点间的最短路径
2. 支持通过中转站点的路径规划
3. 更合理的约束处理和惩罚机制
4. 优化的个体编码和解码策略
"""

import pandas as pd
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ImprovedOilDistributionGA:
    def __init__(self):
        """初始化遗传算法参数和数据"""
        self.population_size = 80
        self.generations = 300
        self.crossover_rate = 0.8
        self.mutation_rate = 0.15
        self.elite_rate = 0.1
        
        # 时间相关参数
        self.start_time = 8.0  # 上午8点
        self.end_time = 17.0   # 下午5点
        self.max_working_hours = self.end_time - self.start_time  # 9小时
        
        # 加载数据
        self.load_data()
        
        # 计算距离矩阵
        self.calculate_distance_matrix()
        
        # 使用Floyd算法计算最短路径
        self.floyd_warshall()
        
        # 预处理需求数据
        self.preprocess_demand()
        
    def load_data(self):
        """加载所有数据文件"""
        try:
            # 加载基础数据
            self.stations_info = pd.read_csv('加油站信息.csv')
            self.stations_demand = pd.read_csv('加油站需求量.csv')
            self.stations_inventory = pd.read_csv('加油站库存.csv')
            self.depot_inventory = pd.read_csv('油库库存.csv')
            self.vehicles_info = pd.read_csv('油罐车信息.csv')
            self.depot_station_distance = pd.read_csv('库站运距.csv')
            self.station_distance = pd.read_csv('站站运距.csv')
            self.oil_info = pd.read_csv('油品信息.csv')
            self.depot_info = pd.read_csv('油库信息.csv')
            
            print("数据加载成功！")
            self.print_data_summary()
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            raise
    
    def print_data_summary(self):
        """打印数据摘要"""
        print("\n=== 数据概览 ===")
        print(f"加油站数量: {len(self.stations_info)}")
        print(f"油罐车数量: {len(self.vehicles_info)}")
        print(f"油品种类: {len(self.oil_info)}")
        print(f"需求记录数: {len(self.stations_demand)}")
        
        # 车辆容量统计
        vehicle_types = self.vehicles_info[['车仓1（升）', '车仓2（升）', '单位距离运输成本']].drop_duplicates()
        print(f"\n车辆类型数: {len(vehicle_types)}")
        for i, (_, row) in enumerate(vehicle_types.iterrows()):
            print(f"  类型{i+1}: 车仓容量 {row['车仓1（升）']}L×2, 成本 {row['单位距离运输成本']}元/km")
    
    def calculate_distance_matrix(self):
        """计算距离矩阵"""
        n_stations = len(self.stations_info)
        n_depots = len(self.depot_info)
        
        # 创建节点编号映射 (油库A=0, 油库B=1, 加油站1-25=2-26)
        self.node_mapping = {}
        self.reverse_mapping = {}
        
        # 油库映射
        for i, depot in self.depot_info.iterrows():
            node_id = i
            self.node_mapping[depot['编码']] = node_id
            self.reverse_mapping[node_id] = depot['编码']
        
        # 加油站映射  
        for i, station in self.stations_info.iterrows():
            node_id = n_depots + i
            self.node_mapping[station['编码']] = node_id
            self.reverse_mapping[node_id] = station['编码']
        
        # 初始化距离矩阵
        total_nodes = n_depots + n_stations
        self.distance_matrix = np.full((total_nodes, total_nodes), np.inf)
        
        # 对角线为0
        np.fill_diagonal(self.distance_matrix, 0)
        
        # 填充油库到加油站的距离
        for _, row in self.depot_station_distance.iterrows():
            depot_id = self.node_mapping[row['油库编码']]
            station_id = self.node_mapping[row['加油站编码']]
            distance = row['运距']
            self.distance_matrix[depot_id][station_id] = distance
            self.distance_matrix[station_id][depot_id] = distance
        
        # 填充加油站之间的距离
        for _, row in self.station_distance.iterrows():
            station1_id = self.node_mapping[row['加油站1编码']]
            station2_id = self.node_mapping[row['加油站2编码']]
            distance = row['运距']
            self.distance_matrix[station1_id][station2_id] = distance
            self.distance_matrix[station2_id][station1_id] = distance
        
        print(f"初始距离矩阵构建完成: {total_nodes}×{total_nodes}")
        
    def floyd_warshall(self):
        """使用Floyd-Warshall算法计算所有节点间的最短路径"""
        print("正在计算最短路径...")
        n = self.distance_matrix.shape[0]
        
        # 复制距离矩阵作为最短路径矩阵
        self.shortest_path_matrix = self.distance_matrix.copy()
        
        # Floyd-Warshall三重循环
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if (self.shortest_path_matrix[i][k] != np.inf and 
                        self.shortest_path_matrix[k][j] != np.inf):
                        new_distance = self.shortest_path_matrix[i][k] + self.shortest_path_matrix[k][j]
                        if new_distance < self.shortest_path_matrix[i][j]:
                            self.shortest_path_matrix[i][j] = new_distance
        
        # 检查是否还有不可达的路径
        unreachable_count = np.sum(self.shortest_path_matrix == np.inf) - n  # 减去对角线
        print(f"Floyd算法完成，剩余不可达路径数: {unreachable_count}")
        
        # 对于仍然不可达的路径，使用一个很大的值代替（相当于通过油库中转）
        max_direct_distance = np.max(self.shortest_path_matrix[self.shortest_path_matrix != np.inf])
        penalty_distance = max_direct_distance * 2.5  # 设置为最大直达距离的2.5倍作为惩罚
        
        self.shortest_path_matrix[self.shortest_path_matrix == np.inf] = penalty_distance
        
        print(f"最短路径计算完成，使用惩罚距离: {penalty_distance:.1f}km")
    
    def preprocess_demand(self):
        """预处理需求数据，计算实际需求量"""
        self.station_demands = {}
        
        for _, row in self.stations_demand.iterrows():
            station_id = self.node_mapping[row['加油站编码']]
            oil_code = row['油品编码']
            
            # 使用最可能需求量作为实际需求
            demand = row['最可能需求量（升）']
            
            if station_id not in self.station_demands:
                self.station_demands[station_id] = {}
            
            self.station_demands[station_id][oil_code] = demand
        
        # 获取库存信息
        self.station_inventories = {}
        for _, row in self.stations_inventory.iterrows():
            station_id = self.node_mapping[row['加油站编码']]
            oil_code = row['油品编码']
            tank_capacity = row['罐容']
            current_inventory = row['库存（升）']
            
            if station_id not in self.station_inventories:
                self.station_inventories[station_id] = {}
            
            if oil_code not in self.station_inventories[station_id]:
                self.station_inventories[station_id][oil_code] = []
            
            self.station_inventories[station_id][oil_code].append({
                'capacity': tank_capacity,
                'current': current_inventory,
                'available': tank_capacity - current_inventory
            })
    
    def calculate_delivery_requirements(self):
        """计算每个加油站的配送需求"""
        self.delivery_requirements = {}
        
        for station_id, demands in self.station_demands.items():
            self.delivery_requirements[station_id] = {}
            
            for oil_code, demand in demands.items():
                if station_id in self.station_inventories and oil_code in self.station_inventories[station_id]:
                    tanks = self.station_inventories[station_id][oil_code]
                    total_available = sum(tank['available'] for tank in tanks)
                    
                    # 计算实际需要配送的量（考虑库存不足的情况）
                    delivery_needed = min(demand, total_available)
                    
                    if delivery_needed > 0:
                        self.delivery_requirements[station_id][oil_code] = delivery_needed
        
        print(f"\n需要配送的加油站数量: {len(self.delivery_requirements)}")
        total_delivery = sum(sum(oils.values()) for oils in self.delivery_requirements.values())
        print(f"总配送量: {total_delivery:,.0f} 升")
    
    def create_individual(self):
        """创建个体（染色体）- 改进版编码方式"""
        individual = []
        remaining_tasks = []
        
        # 收集所有配送任务
        for station_id, oils in self.delivery_requirements.items():
            for oil_code, quantity in oils.items():
                remaining_tasks.append((station_id, oil_code, quantity))
        
        # 随机打乱任务顺序
        random.shuffle(remaining_tasks)
        
        # 优先使用载重量大的车辆
        vehicle_indices = list(range(len(self.vehicles_info)))
        vehicle_indices.sort(key=lambda i: -(self.vehicles_info.iloc[i]['车仓1（升）'] + 
                                            self.vehicles_info.iloc[i]['车仓2（升）']))
        
        for vehicle_idx in vehicle_indices:
            if not remaining_tasks:
                break
                
            vehicle = self.vehicles_info.iloc[vehicle_idx]
            compartment_capacity = vehicle['车仓1（升）']  # 两个储油仓容量相等
            
            # 为储油仓1分配任务
            compartment1_tasks = []
            compartment1_load = 0
            
            # 尝试装载任务到储油仓1
            for task in remaining_tasks[:]:
                station_id, oil_code, quantity = task
                if compartment1_load + quantity <= compartment_capacity:
                    compartment1_tasks.append(task)
                    compartment1_load += quantity
                    remaining_tasks.remove(task)
                    if compartment1_load >= compartment_capacity * 0.8:  # 装载率达到80%就停止
                        break
            
            # 为储油仓2分配任务
            compartment2_tasks = []
            compartment2_load = 0
            
            for task in remaining_tasks[:]:
                station_id, oil_code, quantity = task
                if compartment2_load + quantity <= compartment_capacity:
                    compartment2_tasks.append(task)
                    compartment2_load += quantity
                    remaining_tasks.remove(task)
                    if compartment2_load >= compartment_capacity * 0.8:
                        break
            
            if compartment1_tasks or compartment2_tasks:
                # 生成访问路径（按距离排序优化）
                all_stations = set()
                if compartment1_tasks:
                    all_stations.update([task[0] for task in compartment1_tasks])
                if compartment2_tasks:
                    all_stations.update([task[0] for task in compartment2_tasks])
                
                path = self.optimize_path_order(list(all_stations))
                
                individual.append({
                    'vehicle_id': vehicle_idx,
                    'compartment1': compartment1_tasks,
                    'compartment2': compartment2_tasks,
                    'path': path
                })
        
        return individual
    
    def optimize_path_order(self, stations):
        """使用贪心算法优化访问顺序"""
        if len(stations) <= 1:
            return stations
        
        # 从油库A开始
        depot_a_id = 0
        path = []
        remaining = stations[:]
        current = depot_a_id
        
        while remaining:
            # 找到距离当前位置最近的加油站
            nearest_station = min(remaining, key=lambda s: self.shortest_path_matrix[current][s])
            path.append(nearest_station)
            remaining.remove(nearest_station)
            current = nearest_station
        
        return path
    
    def calculate_fitness(self, individual):
        """计算个体适应度 - 改进版"""
        total_cost = 0
        total_distance = 0
        total_time = 0
        penalty = 0
        
        for gene in individual:
            vehicle_idx = gene['vehicle_id']
            vehicle = self.vehicles_info.iloc[vehicle_idx]
            unit_cost = vehicle['单位距离运输成本']
            path = gene['path']
            
            if not path:
                continue
            
            # 计算路径距离和时间
            route_distance = 0
            route_time = 0
            
            # 从油库A到第一个加油站
            depot_a_id = 0
            first_station = path[0]
            route_distance += self.shortest_path_matrix[depot_a_id][first_station]
            
            # 加油站之间的距离
            for i in range(len(path) - 1):
                from_station = path[i]
                to_station = path[i + 1]
                distance = self.shortest_path_matrix[from_station][to_station]
                route_distance += distance
                
                # 加上卸油时间
                route_time += self.stations_info.iloc[0]['卸油时间']  # 每个站的卸油时间
            
            # 从最后一个加油站返回油库A
            last_station = path[-1]
            route_distance += self.shortest_path_matrix[last_station][depot_a_id]
            
            # 根据距离计算行驶时间
            route_time += route_distance / vehicle['车速（km/hr）']
            
            # 时间窗约束检查
            if route_time > self.max_working_hours:
                penalty += (route_time - self.max_working_hours) * 500
            
            # 计算载重利用率
            compartment_capacity = vehicle['车仓1（升）']
            comp1_load = sum(task[2] for task in gene['compartment1'])
            comp2_load = sum(task[2] for task in gene['compartment2'])
            
            utilization1 = comp1_load / compartment_capacity if compartment_capacity > 0 else 0
            utilization2 = comp2_load / compartment_capacity if compartment_capacity > 0 else 0
            
            # 载重利用率奖励/惩罚
            if utilization1 < 0.3:
                penalty += (0.3 - utilization1) * 1000
            if utilization2 < 0.3:
                penalty += (0.3 - utilization2) * 1000
            
            # 累计成本
            total_cost += route_distance * unit_cost
            total_distance += route_distance
            total_time += route_time
        
        # 检查所有任务是否完成
        completed_tasks = set()
        for gene in individual:
            for task in gene['compartment1'] + gene['compartment2']:
                completed_tasks.add((task[0], task[1]))
        
        required_tasks = set()
        for station_id, oils in self.delivery_requirements.items():
            for oil_code in oils:
                required_tasks.add((station_id, oil_code))
        
        # 未完成任务的惩罚
        uncompleted = len(required_tasks) - len(completed_tasks)
        penalty += uncompleted * 3000
        
        # 车辆使用数量惩罚（鼓励使用更少车辆）
        vehicles_used = len(individual)
        penalty += vehicles_used * 50
        
        # 适应度 = 1 / (总成本 + 惩罚)
        fitness = 1 / (total_cost + penalty + 1)
        
        return fitness, {
            'total_cost': total_cost,
            'total_distance': total_distance,
            'total_time': total_time,
            'penalty': penalty,
            'uncompleted_tasks': uncompleted,
            'vehicles_used': vehicles_used
        }
    
    def crossover(self, parent1, parent2):
        """交叉操作 - 改进版"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # 创建子代
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # 随机选择交叉点
        if len(parent1) > 1 and len(parent2) > 1:
            cross_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
            
            # 交换路径信息
            for i in range(cross_point):
                if i < len(child1) and i < len(parent2):
                    child1[i]['path'] = copy.deepcopy(parent2[i]['path'])
                if i < len(child2) and i < len(parent1):
                    child2[i]['path'] = copy.deepcopy(parent1[i]['path'])
        
        return child1, child2
    
    def mutate(self, individual):
        """变异操作 - 改进版"""
        if random.random() > self.mutation_rate:
            return individual
        
        mutated = copy.deepcopy(individual)
        
        if mutated:
            # 随机选择一个基因进行变异
            gene_idx = random.randint(0, len(mutated) - 1)
            gene = mutated[gene_idx]
            
            # 多种变异策略
            mutation_type = random.choice(['path_swap', 'path_reverse', 'path_optimize'])
            
            if mutation_type == 'path_swap' and len(gene['path']) > 1:
                # 路径交换变异
                i, j = random.sample(range(len(gene['path'])), 2)
                gene['path'][i], gene['path'][j] = gene['path'][j], gene['path'][i]
            
            elif mutation_type == 'path_reverse' and len(gene['path']) > 2:
                # 路径逆转变异
                i = random.randint(0, len(gene['path']) - 2)
                j = random.randint(i + 1, len(gene['path']) - 1)
                gene['path'][i:j+1] = gene['path'][i:j+1][::-1]
            
            elif mutation_type == 'path_optimize':
                # 路径优化变异
                gene['path'] = self.optimize_path_order(gene['path'])
        
        return mutated
    
    def selection(self, population, fitness_scores):
        """选择操作（锦标赛选择）"""
        tournament_size = 3
        selected = []
        
        # 精英保留
        elite_count = int(len(population) * self.elite_rate)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        for idx in elite_indices:
            selected.append(copy.deepcopy(population[idx]))
        
        # 锦标赛选择填充剩余个体
        while len(selected) < len(population):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(copy.deepcopy(population[winner_idx]))
        
        return selected
    
    def run_genetic_algorithm(self):
        """运行遗传算法"""
        print("\n开始计算配送需求...")
        self.calculate_delivery_requirements()
        
        print(f"\n初始化种群大小: {self.population_size}")
        # 初始化种群
        population = []
        for _ in range(self.population_size):
            individual = self.create_individual()
            population.append(individual)
        
        best_fitness_history = []
        avg_fitness_history = []
        best_individual = None
        best_fitness = 0
        
        print(f"\n开始进化，共 {self.generations} 代...")
        
        for generation in range(self.generations):
            # 计算适应度
            fitness_scores = []
            fitness_details = []
            
            for individual in population:
                fitness, details = self.calculate_fitness(individual)
                fitness_scores.append(fitness)
                fitness_details.append(details)
            
            # 记录最佳个体
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_individual = copy.deepcopy(population[max_fitness_idx])
            
            # 记录统计信息
            best_fitness_history.append(max(fitness_scores))
            avg_fitness_history.append(np.mean(fitness_scores))
            
            # 每30代输出进度
            if generation % 30 == 0:
                best_details = fitness_details[max_fitness_idx]
                print(f"第 {generation:3d} 代: 适应度 = {best_fitness:.6f}, "
                      f"成本 = {best_details['total_cost']:.2f}, "
                      f"车辆数 = {best_details['vehicles_used']}, "
                      f"惩罚 = {best_details['penalty']:.0f}")
            
            # 选择
            population = self.selection(population, fitness_scores)
            
            # 交叉
            new_population = []
            for i in range(0, len(population), 2):
                parent1 = population[i]
                parent2 = population[(i + 1) % len(population)]
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([child1, child2])
            
            # 变异
            for i in range(len(new_population)):
                new_population[i] = self.mutate(new_population[i])
            
            population = new_population[:self.population_size]
        
        print(f"\n进化完成！最终最佳适应度: {best_fitness:.6f}")
        
        return best_individual, best_fitness_history, avg_fitness_history
    
    def analyze_solution(self, solution):
        """分析解决方案"""
        if not solution:
            print("无有效解决方案")
            return
        
        print("\n=== 最优配送方案分析 ===")
        
        total_vehicles = len(solution)
        total_distance = 0
        total_cost = 0
        total_delivery = 0
        
        for i, gene in enumerate(solution):
            vehicle_idx = gene['vehicle_id']
            vehicle = self.vehicles_info.iloc[vehicle_idx]
            
            print(f"\n车辆 {i+1} ({vehicle['名称']}):")
            print(f"  车牌: {vehicle['车牌号']}")
            print(f"  储油仓容量: {vehicle['车仓1（升）']}L × 2")
            
            comp1_total = 0
            comp2_total = 0
            
            # 储油仓1任务
            if gene['compartment1']:
                print("  储油仓1任务:")
                for task in gene['compartment1']:
                    station_id, oil_code, quantity = task
                    station_name = f"加油站{station_id-1}站"
                    oil_name = {60001: '92号汽油', 60002: '95号汽油', 60003: '0号柴油'}[oil_code]
                    print(f"    → {station_name}: {oil_name} {quantity:,}L")
                    comp1_total += quantity
                print(f"    储油仓1载量: {comp1_total:,}L ({comp1_total/vehicle['车仓1（升）']*100:.1f}%)")
            
            # 储油仓2任务
            if gene['compartment2']:
                print("  储油仓2任务:")
                for task in gene['compartment2']:
                    station_id, oil_code, quantity = task
                    station_name = f"加油站{station_id-1}站"
                    oil_name = {60001: '92号汽油', 60002: '95号汽油', 60003: '0号柴油'}[oil_code]
                    print(f"    → {station_name}: {oil_name} {quantity:,}L")
                    comp2_total += quantity
                print(f"    储油仓2载量: {comp2_total:,}L ({comp2_total/vehicle['车仓2（升）']*100:.1f}%)")
            
            # 路径信息
            if gene['path']:
                path_str = " → ".join([f"加油站{sid-1}站" for sid in gene['path']])
                print(f"  配送路径: 油库A → {path_str} → 油库A")
                
                # 计算路径距离
                route_distance = 0
                depot_a_id = 0
                
                # 油库到第一站
                route_distance += self.shortest_path_matrix[depot_a_id][gene['path'][0]]
                
                # 站间距离
                for j in range(len(gene['path']) - 1):
                    from_station = gene['path'][j]
                    to_station = gene['path'][j + 1]
                    route_distance += self.shortest_path_matrix[from_station][to_station]
                
                # 最后一站回油库
                route_distance += self.shortest_path_matrix[gene['path'][-1]][depot_a_id]
                
                route_cost = route_distance * vehicle['单位距离运输成本']
                route_time = route_distance / vehicle['车速（km/hr）'] + len(gene['path']) * 0.5
                
                print(f"  路径距离: {route_distance:.1f}km")
                print(f"  预计用时: {route_time:.1f}小时")
                print(f"  运输成本: {route_cost:.2f}元")
                
                total_distance += route_distance
                total_cost += route_cost
                total_delivery += comp1_total + comp2_total
        
        print(f"\n=== 总体统计 ===")
        print(f"使用车辆数: {total_vehicles}")
        print(f"总配送距离: {total_distance:.1f}km")
        print(f"总运输成本: {total_cost:.2f}元")
        print(f"总配送量: {total_delivery:,}L")
        if total_delivery > 0:
            print(f"平均每升配送成本: {total_cost/total_delivery:.4f}元/L")
            print(f"平均每车配送量: {total_delivery/total_vehicles:,.0f}L")
            print(f"平均每车运行距离: {total_distance/total_vehicles:.1f}km")
    
    def plot_convergence(self, best_fitness_history, avg_fitness_history):
        """绘制收敛曲线"""
        plt.figure(figsize=(12, 8))
        
        generations = range(len(best_fitness_history))
        
        plt.subplot(2, 1, 1)
        plt.plot(generations, best_fitness_history, 'b-', linewidth=2, label='最佳适应度')
        plt.plot(generations, avg_fitness_history, 'r--', linewidth=1, label='平均适应度')
        plt.xlabel('进化代数')
        plt.ylabel('适应度')
        plt.title('遗传算法收敛曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 转换为成本曲线（适应度的倒数）
        best_cost = [1/f if f > 0 else float('inf') for f in best_fitness_history]
        avg_cost = [1/f if f > 0 else float('inf') for f in avg_fitness_history]
        
        plt.subplot(2, 1, 2)
        plt.plot(generations, best_cost, 'g-', linewidth=2, label='最佳成本')
        plt.plot(generations, avg_cost, 'm--', linewidth=1, label='平均成本')
        plt.xlabel('进化代数')
        plt.ylabel('成本')
        plt.title('成本变化曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('improved_genetic_algorithm_convergence.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    print("=" * 70)
    print("成品油二次配送车辆路径问题遗传算法求解（改进版）")
    print("支持中转路径，优化约束处理")
    print("=" * 70)
    
    # 创建遗传算法实例
    ga = ImprovedOilDistributionGA()
    
    # 运行遗传算法
    best_solution, best_fitness_history, avg_fitness_history = ga.run_genetic_algorithm()
    
    # 分析最佳解决方案
    ga.analyze_solution(best_solution)
    
    # 绘制收敛曲线
    ga.plot_convergence(best_fitness_history, avg_fitness_history)
    
    print("\n优化完成！")

if __name__ == "__main__":
    main() 