#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成品油配送覆盖率验证和运输方案生成器
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_and_analyze_data():
    """加载数据并分析配送需求"""
    
    # 加载数据
    stations_demand = pd.read_csv('加油站需求量.csv')
    stations_inventory = pd.read_csv('加油站库存.csv')
    vehicles_info = pd.read_csv('油罐车信息.csv')
    depot_info = pd.read_csv('油库信息.csv')
    stations_info = pd.read_csv('加油站信息.csv')
    
    print("=== 配送需求分析 ===")
    
    # 创建节点映射
    node_mapping = {}
    # 油库映射
    for i, depot in depot_info.iterrows():
        node_mapping[depot['编码']] = i
    # 加油站映射
    for i, station in stations_info.iterrows():
        node_mapping[station['编码']] = len(depot_info) + i
    
    # 分析每个加油站的实际配送需求
    delivery_requirements = {}
    
    for _, row in stations_demand.iterrows():
        station_id = node_mapping[row['加油站编码']]
        oil_code = row['油品编码']
        demand = row['最可能需求量（升）']
        
        if station_id not in delivery_requirements:
            delivery_requirements[station_id] = {}
        
        delivery_requirements[station_id][oil_code] = {
            'demand': demand,
            'station_name': row['加油站名称'],
            'oil_name': row['油品名称']
        }
    
    # 检查库存约束
    station_inventories = {}
    for _, row in stations_inventory.iterrows():
        station_id = node_mapping[row['加油站编码']]
        oil_code = row['油品编码']
        available = row['罐容'] - row['库存（升）']
        
        if station_id not in station_inventories:
            station_inventories[station_id] = {}
        if oil_code not in station_inventories[station_id]:
            station_inventories[station_id][oil_code] = []
        
        station_inventories[station_id][oil_code].append(available)
    
    # 计算实际需要配送的量
    final_requirements = {}
    for station_id, demands in delivery_requirements.items():
        final_requirements[station_id] = {}
        
        for oil_code, demand_info in demands.items():
            demand = demand_info['demand']
            if station_id in station_inventories and oil_code in station_inventories[station_id]:
                total_available = sum(station_inventories[station_id][oil_code])
                delivery_needed = min(demand, total_available)
                
                if delivery_needed > 0:
                    final_requirements[station_id][oil_code] = {
                        'delivery_amount': delivery_needed,
                        'station_name': demand_info['station_name'],
                        'oil_name': demand_info['oil_name'],
                        'original_demand': demand,
                        'available_capacity': total_available
                    }
    
    print(f"需要配送的加油站数量: {len(final_requirements)}")
    
    # 详细列出每个站的需求
    total_delivery = 0
    for station_id, oils in final_requirements.items():
        station_name = list(oils.values())[0]['station_name']
        print(f"\n{station_name} (ID: {station_id}):")
        for oil_code, info in oils.items():
            print(f"  {info['oil_name']}: {info['delivery_amount']:,}L (需求: {info['original_demand']:,}L, 可用容量: {info['available_capacity']:,}L)")
            total_delivery += info['delivery_amount']
    
    print(f"\n总配送量: {total_delivery:,}升")
    
    return final_requirements, node_mapping, vehicles_info

def generate_delivery_plan():
    """生成详细的配送方案"""
    
    # 这里是从算法运行结果中提取的最优方案
    # 实际应用中应该从算法输出中读取
    delivery_plan = [
        {
            'vehicle_id': 25,  # 油罐车26
            'vehicle_name': '油罐车26',
            'license_plate': '京MRN767',
            'capacity': [14000, 14000],
            'compartment1': [(25, 60002, 7901), (14, 60001, 5980)],  # 加油站25站95号汽油, 加油站14站92号汽油
            'compartment2': [(16, 60001, 7250), (14, 60002, 6749)],  # 加油站16站92号汽油, 加油站14站95号汽油
            'path': [23, 26, 24],  # 访问加油站编号
            'distance': 125.4,
            'cost': 877.49
        },
        # 可以继续添加更多车辆...
    ]
    
    return delivery_plan

def verify_coverage(final_requirements):
    """验证配送覆盖率"""
    
    print("\n=== 配送覆盖率验证 ===")
    
    # 从算法结果中获取实际配送的任务
    # 这里需要从算法输出中提取实际的配送分配
    
    # 模拟算法输出的配送分配（实际应该从算法结果读取）
    delivered_tasks = set()
    
    # 根据算法运行结果，所有31辆车的配送任务
    algorithm_result_tasks = [
        (25, 60002), (14, 60001), (16, 60001), (14, 60002),  # 车辆1
        (6, 60003), (23, 60002), (7, 60001), (20, 60002),    # 车辆2
        # ... 这里应该包含所有31辆车的配送任务
    ]
    
    # 为了准确验证，我们需要重新运行算法并检查结果
    # 让我们从需求数据重新分析
    
    required_tasks = set()
    for station_id, oils in final_requirements.items():
        for oil_code in oils.keys():
            required_tasks.add((station_id, oil_code))
    
    print(f"需要配送的任务总数: {len(required_tasks)}")
    print("需要配送的任务列表:")
    
    oil_name_map = {60001: '92号汽油', 60002: '95号汽油', 60003: '0号柴油'}
    
    for station_id, oil_code in sorted(required_tasks):
        station_name = f"加油站{station_id-1}站"
        oil_name = oil_name_map[oil_code]
        delivery_amount = final_requirements[station_id][oil_code]['delivery_amount']
        print(f"  {station_name}: {oil_name} {delivery_amount:,}L")
    
    return required_tasks, len(required_tasks)

def create_delivery_schedule_file(final_requirements, vehicles_info):
    """创建详细的配送计划文件"""
    
    # 创建Excel文件包含多个工作表
    with pd.ExcelWriter('成品油配送计划.xlsx', engine='openpyxl') as writer:
        
        # 工作表1：配送需求汇总
        requirements_data = []
        for station_id, oils in final_requirements.items():
            for oil_code, info in oils.items():
                requirements_data.append({
                    '加油站编号': station_id,
                    '加油站名称': info['station_name'],
                    '油品编码': oil_code,
                    '油品名称': info['oil_name'],
                    '配送量(L)': info['delivery_amount'],
                    '原始需求(L)': info['original_demand'],
                    '可用容量(L)': info['available_capacity']
                })
        
        requirements_df = pd.DataFrame(requirements_data)
        requirements_df.to_excel(writer, sheet_name='配送需求汇总', index=False)
        
        # 工作表2：车辆信息
        vehicles_info.to_excel(writer, sheet_name='车辆信息', index=False)
        
        # 工作表3：配送计划模板（需要填入算法结果）
        plan_template = pd.DataFrame({
            '车辆编号': [],
            '车辆名称': [],
            '车牌号': [],
            '储油仓1任务': [],
            '储油仓1载量(L)': [],
            '储油仓1利用率(%)': [],
            '储油仓2任务': [],
            '储油仓2载量(L)': [],
            '储油仓2利用率(%)': [],
            '配送路径': [],
            '路径距离(km)': [],
            '运输成本(元)': [],
            '预计用时(小时)': []
        })
        plan_template.to_excel(writer, sheet_name='配送计划', index=False)
    
    print("\n配送计划文件已生成：成品油配送计划.xlsx")

def main():
    """主函数"""
    print("=" * 60)
    print("成品油配送覆盖率验证和方案生成")
    print("=" * 60)
    
    # 分析配送需求
    final_requirements, node_mapping, vehicles_info = load_and_analyze_data()
    
    # 验证覆盖率
    required_tasks, total_tasks = verify_coverage(final_requirements)
    
    # 生成配送计划文件
    create_delivery_schedule_file(final_requirements, vehicles_info)
    
    # 生成文本版配送计划
    with open('配送计划详细.txt', 'w', encoding='utf-8') as f:
        f.write("成品油二次配送详细计划\n")
        f.write("=" * 50 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("配送需求汇总:\n")
        f.write("-" * 30 + "\n")
        
        total_delivery = 0
        for station_id, oils in final_requirements.items():
            station_name = list(oils.values())[0]['station_name']
            f.write(f"\n{station_name} (ID: {station_id}):\n")
            
            for oil_code, info in oils.items():
                f.write(f"  {info['oil_name']}: {info['delivery_amount']:,}L\n")
                f.write(f"    原始需求: {info['original_demand']:,}L\n")
                f.write(f"    可用容量: {info['available_capacity']:,}L\n")
                total_delivery += info['delivery_amount']
        
        f.write(f"\n总配送量: {total_delivery:,}升\n")
        f.write(f"需要配送的任务数: {total_tasks}个\n")
        
        f.write("\n\n注意事项:\n")
        f.write("-" * 30 + "\n")
        f.write("1. 每辆车有两个储油仓，每个仓的油只能全部卸给一个加油站\n")
        f.write("2. 同一辆车可以同时给多个加油站配送不同油品\n")
        f.write("3. 配送时间窗：上午8点到下午5点\n")
        f.write("4. 支持通过中转站点或油库中转的路径规划\n")
    
    print("\n详细配送计划已生成：配送计划详细.txt")
    
    print(f"\n=== 关键发现 ===")
    print(f"需要配送的加油站数量: {len(final_requirements)}")
    print(f"需要配送的任务总数: {total_tasks}")
    print(f"这表明覆盖率验证需要基于实际的{total_tasks}个配送任务，而不是25个加油站")

if __name__ == "__main__":
    main() 