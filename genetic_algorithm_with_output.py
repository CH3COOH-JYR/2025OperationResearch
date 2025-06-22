#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成品油二次配送车辆路径问题遗传算法解决方案（带详细输出）
支持运输方案文件生成和准确的覆盖率验证
"""

import pandas as pd
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class OilDistributionGAWithOutput:
    def __init__(self):
        """初始化遗传算法参数和数据"""
        self.population_size = 60
        self.generations = 150
        self.crossover_rate = 0.8
        self.mutation_rate = 0.15
        self.elite_rate = 0.1
        
        # 时间相关参数
        self.start_time = 8.0
        self.end_time = 17.0
        self.max_working_hours = self.end_time - self.start_time
        
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
            print(f"加油站数量: {len(self.stations_info)}")
            print(f"油罐车数量: {len(self.vehicles_info)}")
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            raise
    
    def calculate_distance_matrix(self):
        """计算距离矩阵"""
        n_stations = len(self.stations_info)
        n_depots = len(self.depot_info)
        
        # 创建节点编号映射
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
        print("正在计算最短路径（支持中转）...")
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
        unreachable_count = np.sum(self.shortest_path_matrix == np.inf) - n
        print(f"Floyd算法完成，剩余不可达路径数: {unreachable_count}")
        
        # 对于仍然不可达的路径，通过油库中转
        depot_a_id = 0
        for i in range(n):
            for j in range(n):
                if self.shortest_path_matrix[i][j] == np.inf and i != j:
                    # 通过油库A中转的距离
                    transit_distance = (self.shortest_path_matrix[i][depot_a_id] + 
                                      self.shortest_path_matrix[depot_a_id][j])
                    if transit_distance != np.inf:
                        self.shortest_path_matrix[i][j] = transit_distance
                    else:
                        # 如果还是不可达，设置一个很大的惩罚值
                        self.shortest_path_matrix[i][j] = 1000
        
        print("最短路径计算完成，支持油库中转")
    
    def preprocess_demand(self):
        """预处理需求数据"""
        self.station_demands = {}
        
        for _, row in self.stations_demand.iterrows():
            station_id = self.node_mapping[row['加油站编码']]
            oil_code = row['油品编码']
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
                    
                    delivery_needed = min(demand, total_available)
                    
                    if delivery_needed > 0:
                        self.delivery_requirements[station_id][oil_code] = delivery_needed
        
        print(f"需要配送的加油站数量: {len(self.delivery_requirements)}")
        total_delivery = sum(sum(oils.values()) for oils in self.delivery_requirements.values())
        print(f"总配送量: {total_delivery:,.0f} 升")
        
        # 统计总任务数
        self.total_required_tasks = 0
        for station_id, oils in self.delivery_requirements.items():
            self.total_required_tasks += len(oils)
        print(f"需要配送的任务总数: {self.total_required_tasks}")
    
    def create_individual(self):
        """创建个体"""
        individual = []
        remaining_tasks = []
        
        # 收集所有配送任务
        for station_id, oils in self.delivery_requirements.items():
            for oil_code, quantity in oils.items():
                remaining_tasks.append((station_id, oil_code, quantity))
        
        random.shuffle(remaining_tasks)
        
        # 按载重量排序车辆，优先使用大车
        vehicle_indices = list(range(len(self.vehicles_info)))
        vehicle_indices.sort(key=lambda i: -(self.vehicles_info.iloc[i]['车仓1（升）'] + 
                                            self.vehicles_info.iloc[i]['车仓2（升）']))
        
        for vehicle_idx in vehicle_indices:
            if not remaining_tasks:
                break
                
            vehicle = self.vehicles_info.iloc[vehicle_idx]
            compartment_capacity = vehicle['车仓1（升）']
            
            # 为两个储油仓分配任务
            compartment1_tasks = []
            compartment2_tasks = []
            comp1_load = 0
            comp2_load = 0
            
            for task in remaining_tasks[:]:
                station_id, oil_code, quantity = task
                if comp1_load + quantity <= compartment_capacity:
                    compartment1_tasks.append(task)
                    comp1_load += quantity
                    remaining_tasks.remove(task)
                elif comp2_load + quantity <= compartment_capacity:
                    compartment2_tasks.append(task)
                    comp2_load += quantity
                    remaining_tasks.remove(task)
            
            if compartment1_tasks or compartment2_tasks:
                # 优化访问路径
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
        
        depot_a_id = 0
        path = []
        remaining = stations[:]
        current = depot_a_id
        
        while remaining:
            nearest_station = min(remaining, key=lambda s: self.shortest_path_matrix[current][s])
            path.append(nearest_station)
            remaining.remove(nearest_station)
            current = nearest_station
        
        return path
    
    def calculate_fitness(self, individual):
        """计算个体适应度"""
        total_cost = 0
        total_distance = 0
        penalty = 0
        
        for gene in individual:
            vehicle_idx = gene['vehicle_id']
            vehicle = self.vehicles_info.iloc[vehicle_idx]
            unit_cost = vehicle['单位距离运输成本']
            path = gene['path']
            
            if not path:
                continue
            
            route_distance = 0
            depot_a_id = 0
            
            # 计算总路径距离
            route_distance += self.shortest_path_matrix[depot_a_id][path[0]]
            
            for i in range(len(path) - 1):
                route_distance += self.shortest_path_matrix[path[i]][path[i + 1]]
            
            route_distance += self.shortest_path_matrix[path[-1]][depot_a_id]
            
            # 计算时间
            route_time = route_distance / vehicle['车速（km/hr）'] + len(path) * 0.5
            
            # 时间窗约束
            if route_time > self.max_working_hours:
                penalty += (route_time - self.max_working_hours) * 500
            
            # 载重利用率
            compartment_capacity = vehicle['车仓1（升）']
            comp1_load = sum(task[2] for task in gene['compartment1'])
            comp2_load = sum(task[2] for task in gene['compartment2'])
            
            utilization1 = comp1_load / compartment_capacity if compartment_capacity > 0 else 0
            utilization2 = comp2_load / compartment_capacity if compartment_capacity > 0 else 0
            
            # 低利用率惩罚
            if utilization1 < 0.3:
                penalty += (0.3 - utilization1) * 800
            if utilization2 < 0.3:
                penalty += (0.3 - utilization2) * 800
            
            total_cost += route_distance * unit_cost
            total_distance += route_distance
        
        # 检查任务完成情况
        completed_tasks = set()
        for gene in individual:
            for task in gene['compartment1'] + gene['compartment2']:
                completed_tasks.add((task[0], task[1]))
        
        required_tasks = set()
        for station_id, oils in self.delivery_requirements.items():
            for oil_code in oils:
                required_tasks.add((station_id, oil_code))
        
        uncompleted = len(required_tasks) - len(completed_tasks)
        penalty += uncompleted * 2000
        
        fitness = 1 / (total_cost + penalty + 1)
        
        return fitness, {
            'total_cost': total_cost,
            'total_distance': total_distance,
            'penalty': penalty,
            'uncompleted_tasks': uncompleted,
            'vehicles_used': len(individual),
            'completed_tasks': len(completed_tasks),
            'required_tasks': len(required_tasks)
        }
    
    def crossover(self, parent1, parent2):
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        if len(parent1) > 1 and len(parent2) > 1:
            cross_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
            
            for i in range(cross_point):
                if i < len(child1) and i < len(parent2):
                    child1[i]['path'] = copy.deepcopy(parent2[i]['path'])
                if i < len(child2) and i < len(parent1):
                    child2[i]['path'] = copy.deepcopy(parent1[i]['path'])
        
        return child1, child2
    
    def mutate(self, individual):
        """变异操作"""
        if random.random() > self.mutation_rate:
            return individual
        
        mutated = copy.deepcopy(individual)
        
        if mutated:
            gene_idx = random.randint(0, len(mutated) - 1)
            gene = mutated[gene_idx]
            
            if len(gene['path']) > 1:
                i, j = random.sample(range(len(gene['path'])), 2)
                gene['path'][i], gene['path'][j] = gene['path'][j], gene['path'][i]
        
        return mutated
    
    def selection(self, population, fitness_scores):
        """选择操作"""
        tournament_size = 3
        selected = []
        
        # 精英保留
        elite_count = int(len(population) * self.elite_rate)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        for idx in elite_indices:
            selected.append(copy.deepcopy(population[idx]))
        
        # 锦标赛选择
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
        population = []
        for _ in range(self.population_size):
            individual = self.create_individual()
            population.append(individual)
        
        best_fitness_history = []
        best_individual = None
        best_fitness = 0
        
        print(f"\n开始进化，共 {self.generations} 代...")
        
        for generation in range(self.generations):
            fitness_scores = []
            fitness_details = []
            
            for individual in population:
                fitness, details = self.calculate_fitness(individual)
                fitness_scores.append(fitness)
                fitness_details.append(details)
            
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_individual = copy.deepcopy(population[max_fitness_idx])
            
            best_fitness_history.append(max(fitness_scores))
            
            if generation % 15 == 0:
                best_details = fitness_details[max_fitness_idx]
                print(f"第 {generation:3d} 代: 适应度 = {best_fitness:.6f}, "
                      f"成本 = {best_details['total_cost']:.2f}, "
                      f"车辆数 = {best_details['vehicles_used']}, "
                      f"覆盖率 = {best_details['completed_tasks']}/{best_details['required_tasks']} "
                      f"({best_details['completed_tasks']/best_details['required_tasks']*100:.1f}%)")
            
            population = self.selection(population, fitness_scores)
            
            new_population = []
            for i in range(0, len(population), 2):
                parent1 = population[i]
                parent2 = population[(i + 1) % len(population)]
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([child1, child2])
            
            for i in range(len(new_population)):
                new_population[i] = self.mutate(new_population[i])
            
            population = new_population[:self.population_size]
        
        print(f"\n进化完成！最终最佳适应度: {best_fitness:.6f}")
        
        return best_individual, best_fitness_history
    
    def generate_delivery_plan_files(self, solution):
        """生成详细的配送计划文件"""
        
        if not solution:
            print("无有效解决方案，无法生成文件")
            return
        
        # 分析解决方案并计算覆盖率
        completed_tasks = set()
        delivery_plan_data = []
        total_cost = 0
        total_distance = 0
        total_delivery = 0
        
        oil_name_map = {60001: '92号汽油', 60002: '95号汽油', 60003: '0号柴油'}
        
        for i, gene in enumerate(solution):
            vehicle_idx = gene['vehicle_id']
            vehicle = self.vehicles_info.iloc[vehicle_idx]
            
            comp1_total = sum(task[2] for task in gene['compartment1'])
            comp2_total = sum(task[2] for task in gene['compartment2'])
            
            # 计算路径距离
            route_distance = 0
            depot_a_id = 0
            
            if gene['path']:
                route_distance += self.shortest_path_matrix[depot_a_id][gene['path'][0]]
                
                for j in range(len(gene['path']) - 1):
                    from_station = gene['path'][j]
                    to_station = gene['path'][j + 1]
                    route_distance += self.shortest_path_matrix[from_station][to_station]
                
                route_distance += self.shortest_path_matrix[gene['path'][-1]][depot_a_id]
            
            route_cost = route_distance * vehicle['单位距离运输成本']
            route_time = route_distance / vehicle['车速（km/hr）'] + len(gene['path']) * 0.5
            
            # 记录已完成的任务
            for task in gene['compartment1'] + gene['compartment2']:
                completed_tasks.add((task[0], task[1]))
            
            # 准备储油仓任务描述
            comp1_desc = "; ".join([f"加油站{task[0]-1}站-{oil_name_map[task[1]]}-{task[2]}L" 
                                   for task in gene['compartment1']])
            comp2_desc = "; ".join([f"加油站{task[0]-1}站-{oil_name_map[task[1]]}-{task[2]}L" 
                                   for task in gene['compartment2']])
            
            path_desc = " → ".join([f"加油站{sid-1}站" for sid in gene['path']])
            if path_desc:
                path_desc = f"油库A → {path_desc} → 油库A"
            
            delivery_plan_data.append({
                '车辆编号': i + 1,
                '车辆名称': vehicle['名称'],
                '车牌号': vehicle['车牌号'],
                '车仓容量(L)': f"{vehicle['车仓1（升）']} × 2",
                '储油仓1任务': comp1_desc,
                '储油仓1载量(L)': comp1_total,
                '储油仓1利用率(%)': f"{comp1_total/vehicle['车仓1（升）']*100:.1f}%" if vehicle['车仓1（升）'] > 0 else "0%",
                '储油仓2任务': comp2_desc,
                '储油仓2载量(L)': comp2_total,
                '储油仓2利用率(%)': f"{comp2_total/vehicle['车仓2（升）']*100:.1f}%" if vehicle['车仓2（升）'] > 0 else "0%",
                '配送路径': path_desc,
                '路径距离(km)': f"{route_distance:.1f}",
                '运输成本(元)': f"{route_cost:.2f}",
                '预计用时(小时)': f"{route_time:.1f}"
            })
            
            total_cost += route_cost
            total_distance += route_distance
            total_delivery += comp1_total + comp2_total
        
        # 计算覆盖率
        required_tasks = set()
        for station_id, oils in self.delivery_requirements.items():
            for oil_code in oils:
                required_tasks.add((station_id, oil_code))
        
        coverage_rate = len(completed_tasks) / len(required_tasks) * 100
        
        # 生成Excel文件
        with pd.ExcelWriter('详细配送运输方案.xlsx', engine='openpyxl') as writer:
            # 工作表1：配送方案总览
            overview_data = {
                '指标': ['使用车辆数', '总配送距离(km)', '总运输成本(元)', '总配送量(L)', 
                       '平均每升配送成本(元/L)', '任务覆盖率(%)', '完成任务数', '总任务数'],
                '数值': [len(solution), f"{total_distance:.1f}", f"{total_cost:.2f}", 
                        f"{total_delivery:,}", f"{total_cost/total_delivery:.4f}" if total_delivery > 0 else "0",
                        f"{coverage_rate:.1f}%", len(completed_tasks), len(required_tasks)]
            }
            overview_df = pd.DataFrame(overview_data)
            overview_df.to_excel(writer, sheet_name='配送方案总览', index=False)
            
            # 工作表2：详细配送计划
            delivery_df = pd.DataFrame(delivery_plan_data)
            delivery_df.to_excel(writer, sheet_name='详细配送计划', index=False)
            
            # 工作表3：任务完成情况
            task_completion_data = []
            for station_id, oils in self.delivery_requirements.items():
                for oil_code, quantity in oils.items():
                    is_completed = (station_id, oil_code) in completed_tasks
                    task_completion_data.append({
                        '加油站编号': station_id,
                        '加油站名称': f"加油站{station_id-1}站",
                        '油品编码': oil_code,
                        '油品名称': oil_name_map[oil_code],
                        '配送量(L)': quantity,
                        '是否完成': '是' if is_completed else '否'
                    })
            
            task_df = pd.DataFrame(task_completion_data)
            task_df.to_excel(writer, sheet_name='任务完成情况', index=False)
        
        # 生成文本版详细报告
        with open('配送运输方案详细报告.txt', 'w', encoding='utf-8') as f:
            f.write("成品油二次配送运输方案详细报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("方案总览:\n")
            f.write("-" * 30 + "\n")
            f.write(f"使用车辆数: {len(solution)}辆\n")
            f.write(f"总配送距离: {total_distance:.1f}km\n")
            f.write(f"总运输成本: {total_cost:.2f}元\n")
            f.write(f"总配送量: {total_delivery:,}升\n")
            f.write(f"平均每升配送成本: {total_cost/total_delivery:.4f}元/L\n")
            f.write(f"任务覆盖率: {coverage_rate:.1f}% ({len(completed_tasks)}/{len(required_tasks)})\n\n")
            
            f.write("详细配送计划:\n")
            f.write("-" * 30 + "\n")
            for data in delivery_plan_data:
                f.write(f"\n车辆 {data['车辆编号']} ({data['车辆名称']}):\n")
                f.write(f"  车牌: {data['车牌号']}\n")
                f.write(f"  车仓容量: {data['车仓容量(L)']}\n")
                if data['储油仓1任务']:
                    f.write(f"  储油仓1: {data['储油仓1任务']} (载量: {data['储油仓1载量(L)']}L, 利用率: {data['储油仓1利用率(%)']})\n")
                if data['储油仓2任务']:
                    f.write(f"  储油仓2: {data['储油仓2任务']} (载量: {data['储油仓2载量(L)']}L, 利用率: {data['储油仓2利用率(%)']})\n")
                f.write(f"  配送路径: {data['配送路径']}\n")
                f.write(f"  路径距离: {data['路径距离(km)']}km\n")
                f.write(f"  运输成本: {data['运输成本(元)']}元\n")
                f.write(f"  预计用时: {data['预计用时(小时)']}小时\n")
            
            # 未完成任务列表
            uncompleted_tasks = required_tasks - completed_tasks
            if uncompleted_tasks:
                f.write(f"\n未完成任务列表 (共{len(uncompleted_tasks)}个):\n")
                f.write("-" * 30 + "\n")
                for station_id, oil_code in sorted(uncompleted_tasks):
                    station_name = f"加油站{station_id-1}站"
                    oil_name = oil_name_map[oil_code]
                    quantity = self.delivery_requirements[station_id][oil_code]
                    f.write(f"  {station_name}: {oil_name} {quantity:,}L\n")
        
        print(f"\n=== 配送方案文件生成完成 ===")
        print(f"1. Excel文件: 详细配送运输方案.xlsx")
        print(f"2. 文本报告: 配送运输方案详细报告.txt")
        print(f"\n=== 覆盖率验证结果 ===")
        print(f"总任务数: {len(required_tasks)}")
        print(f"完成任务数: {len(completed_tasks)}")
        print(f"覆盖率: {coverage_rate:.1f}%")
        
        if coverage_rate < 100:
            print(f"注意: 有 {len(uncompleted_tasks)} 个任务未完成，覆盖率不是100%")
        else:
            print("✓ 实现100%覆盖率！")

def main():
    """主函数"""
    print("=" * 70)
    print("成品油二次配送车辆路径问题遗传算法求解（带详细输出）")
    print("支持运输方案文件生成和准确的覆盖率验证")
    print("=" * 70)
    
    # 创建遗传算法实例
    ga = OilDistributionGAWithOutput()
    
    # 运行遗传算法
    best_solution, best_fitness_history = ga.run_genetic_algorithm()
    
    # 生成详细的配送计划文件
    ga.generate_delivery_plan_files(best_solution)
    
    print("\n优化完成！")

if __name__ == "__main__":
    main() 