
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Ihab S. Mohamed, Vehicle Autonomy and Intelligence Lab - Indiana University, Bloomington, USA
"""
import yaml
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib import ticker
from matplotlib.animation import FuncAnimation, ImageMagickWriter
import matplotlib.patches as patches
from pathlib import Path
import matplotlib.cm as cm
import seaborn as sns
from casadi import *

"""  绘制 路径图 """
def pathPlotting(ob_num_max, safety_dist, robot_r, targets ,results_rootpath):

    # 读取CSV文件
    # robot_1_ = pd.read_csv(results_rootpath + '/N-DC.csv')
    robot_2_ = pd.read_csv(results_rootpath + '/N-DCBF.csv')
    # robot_3_ = pd.read_csv(results_rootpath + '/N-RCBF.csv')
    # robot_4_ = pd.read_csv(results_rootpath + '/LOG-RCBF.csv')    
    robot_5_ = pd.read_csv(results_rootpath + '/COAA-RCBF.csv')  
    ob_ = pd.read_csv(results_rootpath + '/obs_results.csv')

    states = []
    robot_size = len(robot_2_)

    ob_size = len(ob_)
    if robot_size >= ob_size:
        size = ob_size
    else:
        size = robot_size
    # 如果数据长度大于 max_num，适当跳过一些数据
    max_num = 300
    if size > max_num:
        robot_indices = np.linspace(0, robot_size - 1, max_num).astype(int)
        # robot_1 = robot_1_.iloc[robot_indices].reset_index(drop=True)  # 重采样后的障碍物数据
        robot_2 = robot_2_.iloc[robot_indices].reset_index(drop=True)  # 重采样后的障碍物数据
        # robot_3 = robot_3_.iloc[robot_indices].reset_index(drop=True)  # 重采样后的障碍物数据
        # robot_4 = robot_4_.iloc[robot_indices].reset_index(drop=True)  # 重采样后的障碍物数据
        robot_5 = robot_5_.iloc[robot_indices].reset_index(drop=True)  # 重采样后的障碍物数据
    # 为 ob 数据选择均匀的 max_num 个点
        ob_indices = np.linspace(0, ob_size - 1, max_num).astype(int)
        ob = ob_.iloc[ob_indices].reset_index(drop=True)  # 重采样后的障碍物数据
    else:
        # robot_1 = robot_1_
        robot_2 = robot_2_
        # robot_3 = robot_3_
        # robot_4 = robot_4_
        robot_5 = robot_5_
        ob = ob_

    """Plots the robot path in the x-y plane."""
    # sns.set_theme()

    # # 全局背景设置
    # plt.rcParams.update({
    #     'figure.facecolor': 'white',  # 画布背景
    #     'axes.facecolor': 'white',     # 坐标轴区域背景
    #     'savefig.facecolor': 'white',   # 保存图片时的背景
    #     'axes.grid': False             # 彻底关闭网格线
    # })

    # # 设置seaborn主题为纯白（无网格）
    # sns.set_theme(style="white")  # 关键修改：从whitegrid改为white

    # 全局设置（背景保持白色，网格参数调整）
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.grid': True,               # 开启网格线（<a target="_blank" href="https://www.cnblogs.com/ivanlee717/p/17483909.html" class="hitref" data-title="matplot画网格线分层级的用法- ivanlee717 - 博客园" data-snippet='Matplotlib 的 grid() 函数可以用于绘制网格线。 该函数的常用参数如下： - b 设置是否显示网格线。 可以取值为True 或False。' data-url="https://www.cnblogs.com/ivanlee717/p/17483909.html">1</a><a target="_blank" href="http://www.runoob.com/matplotlib/matplotlib-grid.html" class="hitref" data-title="Matplotlib 网格线 - 菜鸟教程" data-snippet='Matplotlib 网格线我们可以使用pyplot 中的grid() 方法来设置图表中的网格线。 grid() 方法语法格式如下： matplotlib.pyplot.grid(b=None, which=&#39;major&#39;, axis=&#39;both&#39; ...' data-url="http://www.runoob.com/matplotlib/matplotlib-grid.html">7</a>）
        'grid.color': '#e0e0e0',         # 浅灰色（HEX格式，<a target="_blank" href="https://wenku.csdn.net/answer/63f85veryr" class="hitref" data-title="matplotlib 浅灰色 - CSDN文库" data-snippet='要设置matplotlib绘图的浅灰色，可以使用RGB颜色代码表示。以下是一个示例代码，用于设置浅灰色背景： import matplotlib.pyplot as plt # 设置浅灰色RGB ...' data-url="https://wenku.csdn.net/answer/63f85veryr">4</a><a target="_blank" href="http://www.runoob.com/matplotlib/matplotlib-grid.html" class="hitref" data-title="Matplotlib 网格线 - 菜鸟教程" data-snippet='Matplotlib 网格线我们可以使用pyplot 中的grid() 方法来设置图表中的网格线。 grid() 方法语法格式如下： matplotlib.pyplot.grid(b=None, which=&#39;major&#39;, axis=&#39;both&#39; ...' data-url="http://www.runoob.com/matplotlib/matplotlib-grid.html">7</a>）
        'grid.alpha': 0.3,               # 透明度（<a target="_blank" href="https://www.python91.com/python_Matplotlib/187.html" class="hitref" data-title="matplotlib中的grid()方法如何设置网格线外观 - python编程" data-snippet='设置透明度：可以使用alpha参数来设置网格线的透明度，例如grid(alpha=0.5)可以将网格线的透明度设置为0.5。 综合使用这些参数可以实现各种网格线的外观设置 ...' data-url="https://www.python91.com/python_Matplotlib/187.html">10</a>）
        'grid.linestyle': '--',          # 虚线样式（可选）
        'axes.axisbelow': True           # 网格线在数据下方（<a target="_blank" href="https://www.osgeo.cn/matplotlib/users/dflt_style_changes.html" class="hitref" data-title="更改为默认样式— Matplotlib 3.3.3 文档" data-snippet='记号和网格现在绘制在实体元素（如填充轮廓）的上方，但绘制在线的下方。若要返回到上一个绘制线上方刻度和网格的行为，请设置 rcParams[&#39;axes.axisbelow&#39;] = False .' data-url="https://www.osgeo.cn/matplotlib/users/dflt_style_changes.html">6</a>）
    })

    # 保持seaborn主题但覆盖网格颜色（可选）
    sns.set_theme(style="whitegrid", rc={'grid.color': '#e0e0e0', 'grid.alpha': 0.3})

    fig, ax = plt.subplots(figsize=(5, 5))
    # ax.plot(robot_1['x'][:], robot_1['y'][:], color='b', label="N-DC")
    ax.plot(robot_2['x'][:], robot_2['y'][:], color='r', label="N-DCBF")
    # ax.plot(robot_3['x'][:], robot_3['y'][:], color='g', label="N-RCBF")
    # ax.plot(robot_4['x'][:], robot_4['y'][:], color='y', label="L-RCBF")
    ax.plot(robot_5['x'][:], robot_5['y'][:], color='c', label="C-RCBF\n(proposed)")

    ax.set_xlabel('x position [m]', fontsize=12, labelpad=0)
    ax.set_ylabel('y position [m]', fontsize=12, labelpad=0)
    ax.tick_params(axis='both', labelsize=10, colors='#444444')
    # plt.title("Robot path")
    plt.tight_layout()


    plt.xlim(-1.2, targets[0]+1.2)
    plt.ylim(-6.2, targets[1]+6.2)
    ax.set_aspect('equal', adjustable='box')  # 保持比例同时锁定坐标范围

    # 机器人初始位置
    ax.plot(0.0, 0.0, 'r.', label="Initial position")
    # ax.plot(0.0, 0.0, 'r.')
    # 机器人终点位置
    # ax.add_patch(plt.Circle((robot_1['x'].iloc[-1], robot_1['y'].iloc[-1]), robot_r, color='b', zorder=2))
    # ax.add_patch(plt.Circle((robot_2['x'].iloc[-1], robot_2['y'].iloc[-1]), robot_r, color='r', zorder=2))
    # ax.add_patch(plt.Circle((robot_3['x'].iloc[-1], robot_3['y'].iloc[-1]), robot_r, color='g', zorder=2))


    ax.plot(targets[0], targets[1], 'g*', 
            markersize=15,  # 调整此处数值控制大小
            markeredgewidth=1.2,  # 边缘线宽（可选）
            label="Goal")

    for id in range(ob_num_max):
        # 绘制初始障碍物位置
        circle = plt.Circle([ob[f'x{id}'][0],ob[f'y{id}'][0]], 
                                    ob[f'r{id}'][0], 
                                        color='#2E4053', # 颜色为灰色
                                            fill=True,  # 填充颜色
                                                # linestyle='--', # 虚线
                                                    linewidth=2, # 边界宽度
                                                        alpha=0.2) # 透明度
        plt.gca().add_artist(circle)

        circle_ = plt.Circle([ob[f'x{id}'][0],ob[f'y{id}'][0]],
                                ob[f'r{id}'][0] + safety_dist + robot_r, 
                                    edgecolor='#6C3483',  # 边界颜色为红色
                                        facecolor='none',  # 内部无颜色
                                            linestyle='--',  # 虚线
                                                linewidth=1,  # 边界宽度
                                                    alpha=0.2)  # 透明度
        plt.gca().add_artist(circle_)    


        # # 绘制障碍物轨迹
        # ax.plot(ob[f'x{id}'][:], ob[f'y{id}'][:],
        #         'k:', label="Obstacle path", alpha=0.3)
        
        # 在障碍物轨迹绘制部分替换为：
        # 创建紫色渐变colormap（浅紫到深紫）
        # purple_cmap = LinearSegmentedColormap.from_list(
        #     'purple_gradient', 
        #     ['#DDCCFF', '#330066']  # 浅紫 -> 深紫
        # )

        # # 绘制带图例的初始点（仅用于标签）
        # path = ax.plot(ob[f'x{id}'][0:1], ob[f'y{id}'][0:1], 
        #             color=purple_cmap(0),  # 使用最浅紫色
        #             linestyle=':', 
        #             alpha=0.3, 
        #             label="Obstacle path")[0]

        # # 逐段绘制渐变路径
        # for i in range(1, len(ob[f'x{id}'])):
        #     t = i / len(ob[f'x{id}'])  # 归一化位置
        #     ax.plot(ob[f'x{id}'][i-1:i+1], 
        #             ob[f'y{id}'][i-1:i+1],
        #             color=purple_cmap(t),  # 动态颜色
        #             linestyle=':', 
        #             linewidth=2.2,  # 稍粗线宽
        #             alpha=0.6,
        #             zorder=1)

        # 在障碍物轨迹绘制部分替换为：
        # 创建灰度渐变colormap（从浅灰到黑）
        # gray_cmap = LinearSegmentedColormap.from_list('gray_gradient', ['#AEB6BF', '#000000'])

        # # 绘制渐变轨迹
        # path = ax.plot(ob[f'x{id}'][0:1], ob[f'y{id}'][0:1],  # 初始点用于创建图例
        #             color='#DDDDDD', linestyle=':', alpha=0.3, label="Obstacle path")[0]

        # # 逐段绘制渐变路径
        # for i in range(1, len(ob[f'x{id}'])):
        #     # 计算颜色插值（0~1之间的归一化位置）
        #     t = i / len(ob[f'x{id}'])
        #     color = gray_cmap(t)
            
        #     # 绘制单段路径
        #     ax.plot(ob[f'x{id}'][i-1:i+1], 
        #             ob[f'y{id}'][i-1:i+1],
        #             color=color, 
        #             linestyle=':', 
        #             linewidth=2.2,
        #             alpha=0.5,
        #             zorder=1)

                # 计算每个时间步的距离
        distances = np.sqrt(
            (robot_5['x'] - ob[f'x{id}'])**2 + 
            (robot_5['y'] - ob[f'y{id}'])**2
        )
        # 找到最小距离的索引
        t_min = distances.idxmin()

        # 在轨迹绘制部分修改为：
        # 计算终止索引（不超过数据长度）
        end_idx = min(t_min + 20, len(ob[f'x{id}'])-1)

        # 创建渐变映射（基于实际绘制长度）
        gray_cmap = LinearSegmentedColormap.from_list(
            'gray_gradient', ['#AEB6BF', '#000000'], N=end_idx
        )

        # 绘制渐变轨迹（仅到end_idx）
        for i in range(1, end_idx+1):
            t = i / end_idx  # 归一化到实际绘制长度
            ax.plot(ob[f'x{id}'][i-1:i+1], 
                    ob[f'y{id}'][i-1:i+1],
                    color=gray_cmap(t), 
                    linestyle=':', 
                    linewidth=2.2,
                    alpha=0.5,
                    zorder=1)

        # 在轨迹末端添加方向箭头
        if end_idx > 1:
            # 计算最后两个点的方向向量
            dx = ob[f'x{id}'][end_idx] - ob[f'x{id}'][end_idx-1]
            dy = ob[f'y{id}'][end_idx] - ob[f'y{id}'][end_idx-1]
            
            # 绘制箭头（缩放系数0.8使箭头更紧凑）
            ax.arrow(ob[f'x{id}'][end_idx-1], 
                    ob[f'y{id}'][end_idx-1],
                    # dx*0.8, dy*0.8,
                    -0.3, -0.0,
                    head_width=0.3,
                    head_length=0.5,
                    fc=gray_cmap(1.0),  # 使用最深颜色
                    ec=gray_cmap(1.0),
                    linewidth=1.8,
                    alpha=0.7,
                    zorder=2)



        # 获取障碍物属性
        x_ob = ob[f'x{id}'][t_min]
        y_ob = ob[f'y{id}'][t_min]
        r_ob = ob[f'r{id}'][t_min]

        # 绘制 最小距离 的 障碍物位置
        circle = plt.Circle([x_ob,y_ob], 
                                    r_ob, 
                                        color='#2E4053', # 颜色为灰色
                                            fill=True,  # 填充颜色
                                                # linestyle='--', # 虚线
                                                    linewidth=1, # 边界宽度
                                                        alpha=0.6) # 透明度
        plt.gca().add_artist(circle)

        circle_ = plt.Circle([x_ob,y_ob],
                                r_ob + safety_dist + robot_r, 
                                    edgecolor='#6C3483',  # 边界颜色为红色
                                        facecolor='none',  # 内部无颜色
                                            linestyle='--',  # 虚线
                                                linewidth=1,  # 边界宽度
                                                    alpha=0.6)  # 透明度
        plt.gca().add_artist(circle_)    

        # 计算速度方向（基于位置变化）
        if t_min < len(ob) - 1:
            vx = ob[f'x{id}'][t_min+1] - ob[f'x{id}'][t_min]
            vy = ob[f'y{id}'][t_min+1] - ob[f'y{id}'][t_min]
        else:
            vx = ob[f'x{id}'][t_min] - ob[f'x{id}'][t_min-1]
            vy = ob[f'y{id}'][t_min] - ob[f'y{id}'][t_min-1]

        # 归一化速度方向（保持箭头长度一致）
        norm = np.hypot(vx, vy)
        if norm > 0:
            vx /= norm
            vy /= norm

        # 修改速度箭头部分为：
        ax.arrow(
            x_ob, y_ob, 
            # vx*0.5, vy*0.5,
            -0.2, -0.0,
            head_width=0.25, 
            color='#CD5C5C',     # 橙色箭头
            linewidth=1.5,      # 加粗线宽
            alpha=0.9,          # 提高透明度
            zorder=4,
            head_starts_at_zero=True  # 箭头从起点开始
        )

        # ax.arrow(
        #     x_ob, y_ob, vx*0.3, vy*0.3,
        #     head_width=0.25, 
        #     color='#9C640C',     # 橙色箭头
        #     linewidth=1.0,      # 加粗线宽
        #     alpha=0.6,          # 提高透明度
        #     zorder=4,
        #     head_starts_at_zero=True  # 箭头从起点开始
        # )

    legend_elements = [
        Line2D([0], [0], 
            color='#6C3483', 
            lw=1.5,
            linestyle='--',
            markersize=10,      # 标记大小
            markeredgewidth=0,
            label='Valid radius'),             
        # 渐变轨迹图示
        Line2D([0], [0],
            color='#808080',  # 中间灰色
            linestyle=':',
            linewidth=2.2,
            marker='>',       # 末端箭头
            markersize=10,
            markeredgecolor='#404040',
            markerfacecolor='#202020',
            alpha=0.7,
            label='Path'),              
        Line2D([0], [0], 
            color='#CD5C5C', 
            lw=1.5,
            marker='>',         # 添加箭头标记
            markersize=10,      # 标记大小
            markeredgewidth=0,
            label='Acceleration'),   
                 
        # Line2D([0], [0],
        #     color='#9C640C',
        #     lw=1.0,
        #     marker='>',
        #     markersize=8,
        #     markeredgewidth=0,
        #     label='Speed')
    ]
    # 创建第一个图例（原有元素，右下角）    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  
    original_legend = ax.legend(
        handles=by_label.values(),
        labels=by_label.keys(),
        loc='upper left',
        framealpha=0.9,
        title_fontsize=10,
        borderpad=0.3         # 图例内边距
    )

    # # 合并原有图例和新图例
    # ax.legend(handles=legend_elements,  # 合并原有图例项
    #         # labels=['Speed', 'ACC'],
    #         labels=['Obstacle Acceleration', 'Obstacle Path'],
    #         loc='upper left', 
    #         framealpha=0.9)

    ax.add_artist(original_legend)  # 必须保留原有图例对象
    ax.legend(
        handles=legend_elements,
        loc='lower right',
        framealpha=0.9,
        title='Obstacles',
        title_fontsize=10,
        borderpad=0.3         # 图例内边距
    )

    ax.set_facecolor('white')  # 设置坐标轴背景为白色


    # Only show unique legends
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), loc="lower right")

    save_path = results_rootpath + '/co_path2.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()

"""  绘制 路径图 """
def timePlotting(results_rootpath):

    # 读取CSV文件
    robot_2_ = pd.read_csv(results_rootpath + '/N-DCBF-T.csv')   
    robot_5_ = pd.read_csv(results_rootpath + '/COAA-RCBF-T.csv')  

    robot_size2 = len(robot_2_)

    robot_size5 = len(robot_5_)
    if robot_size2 >= robot_size5:
        size = robot_size2
    else:
        size = robot_size5
    # 如果数据长度大于 max_num，适当跳过一些数据
    max_num = 300
    if size > max_num:
        robot_indices = np.linspace(0, size - 1, max_num).astype(int)
        robot_2 = robot_2_.iloc[robot_indices].reset_index(drop=True)  # 重采样后的障碍物数据
        robot_5 = robot_5_.iloc[robot_indices].reset_index(drop=True)  # 重采样后的障碍物数据
        size = max_num
    else:
        robot_2 = robot_2_
        robot_5 = robot_5_

    # Ensure T_2 matches the length of robot_2['t_mppi']
    T_2 = np.arange(0, 0.1 * robot_size2, 0.1)[:len(robot_2['t_mppi'])]
    T_5 = np.arange(0, 0.1 * robot_size5, 0.1)[:len(robot_5['t_mppi'])]

    # 全局设置（背景保持白色，网格参数调整）
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.grid': True,               # 开启网格线（<a target="_blank" href="https://www.cnblogs.com/ivanlee717/p/17483909.html" class="hitref" data-title="matplot画网格线分层级的用法- ivanlee717 - 博客园" data-snippet='Matplotlib 的 grid() 函数可以用于绘制网格线。 该函数的常用参数如下： - b 设置是否显示网格线。 可以取值为True 或False。' data-url="https://www.cnblogs.com/ivanlee717/p/17483909.html">1</a><a target="_blank" href="http://www.runoob.com/matplotlib/matplotlib-grid.html" class="hitref" data-title="Matplotlib 网格线 - 菜鸟教程" data-snippet='Matplotlib 网格线我们可以使用pyplot 中的grid() 方法来设置图表中的网格线。 grid() 方法语法格式如下： matplotlib.pyplot.grid(b=None, which=&#39;major&#39;, axis=&#39;both&#39; ...' data-url="http://www.runoob.com/matplotlib/matplotlib-grid.html">7</a>）
        'grid.color': '#e0e0e0',         # 浅灰色（HEX格式，<a target="_blank" href="https://wenku.csdn.net/answer/63f85veryr" class="hitref" data-title="matplotlib 浅灰色 - CSDN文库" data-snippet='要设置matplotlib绘图的浅灰色，可以使用RGB颜色代码表示。以下是一个示例代码，用于设置浅灰色背景： import matplotlib.pyplot as plt # 设置浅灰色RGB ...' data-url="https://wenku.csdn.net/answer/63f85veryr">4</a><a target="_blank" href="http://www.runoob.com/matplotlib/matplotlib-grid.html" class="hitref" data-title="Matplotlib 网格线 - 菜鸟教程" data-snippet='Matplotlib 网格线我们可以使用pyplot 中的grid() 方法来设置图表中的网格线。 grid() 方法语法格式如下： matplotlib.pyplot.grid(b=None, which=&#39;major&#39;, axis=&#39;both&#39; ...' data-url="http://www.runoob.com/matplotlib/matplotlib-grid.html">7</a>）
        'grid.alpha': 0.3,               # 透明度（<a target="_blank" href="https://www.python91.com/python_Matplotlib/187.html" class="hitref" data-title="matplotlib中的grid()方法如何设置网格线外观 - python编程" data-snippet='设置透明度：可以使用alpha参数来设置网格线的透明度，例如grid(alpha=0.5)可以将网格线的透明度设置为0.5。 综合使用这些参数可以实现各种网格线的外观设置 ...' data-url="https://www.python91.com/python_Matplotlib/187.html">10</a>）
        'grid.linestyle': '--',          # 虚线样式（可选）
        'axes.axisbelow': True           # 网格线在数据下方（<a target="_blank" href="https://www.osgeo.cn/matplotlib/users/dflt_style_changes.html" class="hitref" data-title="更改为默认样式— Matplotlib 3.3.3 文档" data-snippet='记号和网格现在绘制在实体元素（如填充轮廓）的上方，但绘制在线的下方。若要返回到上一个绘制线上方刻度和网格的行为，请设置 rcParams[&#39;axes.axisbelow&#39;] = False .' data-url="https://www.osgeo.cn/matplotlib/users/dflt_style_changes.html">6</a>）
    })

    # 保持seaborn主题但覆盖网格颜色（可选）
    sns.set_theme(style="whitegrid", rc={'grid.color': '#e0e0e0', 'grid.alpha': 0.3})

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(T_2[1:], robot_2['t_mppi'][1:], color='r', label="N-DCBF")
    ax.plot(T_5[1:], robot_5['t_mppi'][1:], color='c', label="C-RCBF\n(proposed)")

    ax.set_xlabel('time [s]', fontsize=12, labelpad=0)
    ax.set_ylabel('solving time [ms]', fontsize=12, labelpad=0)
    ax.tick_params(axis='both', labelsize=10, colors='#444444')
    plt.tight_layout()

    # 获取除了第一个元素外的所有元素
    values2 = robot_2['t_mppi'][1:]

    # 计算这些元素的和并求平均值
    mean_value2 = values2.sum() / len(values2)

    # 获取除了第一个元素外的所有元素
    values5 = robot_5['t_mppi'][1:]

    # 计算这些元素的和并求平均值
    mean_value5 = values5.sum() / len(values5)
    print(f'N-DCBF: {mean_value2} ms')
    print(f'C-RCBF: {mean_value5} ms')
    plt.axhline(y=mean_value2, color='r', linestyle='--')
    plt.axhline(y=mean_value5, color='c', linestyle='--')

    plt.xlim(-0.2, T_2[-1]+0.2)
    plt.ylim(1.2, 4.2)
    # ax.set_aspect('equal', adjustable='box')  # 保持比例同时锁定坐标范围

    # 创建第一个图例（原有元素，右下角）    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  
    original_legend = ax.legend(
        handles=by_label.values(),
        labels=by_label.keys(),
        loc='upper left',
        framealpha=0.9,
        title_fontsize=10,
        borderpad=0.3         # 图例内边距
    )

    ax.add_artist(original_legend)  # 必须保留原有图例对象

    ax.set_facecolor('white')  # 设置坐标轴背景为白色

    save_path = results_rootpath + '/co_time2.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()    

def plot_cbf(dt, ob_num_max, robot_r, safety_dist, results_rootpath):
    """Plots the CBF values."""

    # 读取CSV文件
    robot_2_ = pd.read_csv(results_rootpath + '/N-DCBF.csv')   
    robot_5_ = pd.read_csv(results_rootpath + '/COAA-RCBF.csv')  
    ob_ = pd.read_csv(results_rootpath + '/obs_results.csv')
    
    robot_size2 = len(robot_2_)

    robot_size5 = len(robot_5_)
    if robot_size2 >= robot_size5:
        size = robot_size2
    else:
        size = robot_size5
    # 如果数据长度大于 max_num，适当跳过一些数据
    max_num = 300
    if size > max_num:
        robot_indices = np.linspace(0, size - 1, max_num).astype(int)
        robot_2 = robot_2_.iloc[robot_indices].reset_index(drop=True)  # 重采样后的障碍物数据
        robot_5 = robot_5_.iloc[robot_indices].reset_index(drop=True)  # 重采样后的障碍物数据
        size = max_num
    else:
        robot_2 = robot_2_
        robot_5 = robot_5_
        ob = ob_
    # Ensure T_2 matches the length of robot_2['t_mppi']
    T_2 = np.arange(0, 0.1 * robot_size2, 0.1)[:len(robot_2['t_mppi'])]
    T_5 = np.arange(0, 0.1 * robot_size5, 0.1)[:len(robot_5['t_mppi'])]


    def cbf_h(s, obs, robot_r):
        h = sqrt((s[0]-obs[0])**2 + (s[1]-obs[1])**2) - (robot_r + obs[2])
        return h


    # 全局设置（背景保持白色，网格参数调整）
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.grid': True,               # 开启网格线（<a target="_blank" href="https://www.cnblogs.com/ivanlee717/p/17483909.html" class="hitref" data-title="matplot画网格线分层级的用法- ivanlee717 - 博客园" data-snippet='Matplotlib 的 grid() 函数可以用于绘制网格线。 该函数的常用参数如下： - b 设置是否显示网格线。 可以取值为True 或False。' data-url="https://www.cnblogs.com/ivanlee717/p/17483909.html">1</a><a target="_blank" href="http://www.runoob.com/matplotlib/matplotlib-grid.html" class="hitref" data-title="Matplotlib 网格线 - 菜鸟教程" data-snippet='Matplotlib 网格线我们可以使用pyplot 中的grid() 方法来设置图表中的网格线。 grid() 方法语法格式如下： matplotlib.pyplot.grid(b=None, which=&#39;major&#39;, axis=&#39;both&#39; ...' data-url="http://www.runoob.com/matplotlib/matplotlib-grid.html">7</a>）
        'grid.color': '#e0e0e0',         # 浅灰色（HEX格式，<a target="_blank" href="https://wenku.csdn.net/answer/63f85veryr" class="hitref" data-title="matplotlib 浅灰色 - CSDN文库" data-snippet='要设置matplotlib绘图的浅灰色，可以使用RGB颜色代码表示。以下是一个示例代码，用于设置浅灰色背景： import matplotlib.pyplot as plt # 设置浅灰色RGB ...' data-url="https://wenku.csdn.net/answer/63f85veryr">4</a><a target="_blank" href="http://www.runoob.com/matplotlib/matplotlib-grid.html" class="hitref" data-title="Matplotlib 网格线 - 菜鸟教程" data-snippet='Matplotlib 网格线我们可以使用pyplot 中的grid() 方法来设置图表中的网格线。 grid() 方法语法格式如下： matplotlib.pyplot.grid(b=None, which=&#39;major&#39;, axis=&#39;both&#39; ...' data-url="http://www.runoob.com/matplotlib/matplotlib-grid.html">7</a>）
        'grid.alpha': 0.3,               # 透明度（<a target="_blank" href="https://www.python91.com/python_Matplotlib/187.html" class="hitref" data-title="matplotlib中的grid()方法如何设置网格线外观 - python编程" data-snippet='设置透明度：可以使用alpha参数来设置网格线的透明度，例如grid(alpha=0.5)可以将网格线的透明度设置为0.5。 综合使用这些参数可以实现各种网格线的外观设置 ...' data-url="https://www.python91.com/python_Matplotlib/187.html">10</a>）
        'grid.linestyle': '--',          # 虚线样式（可选）
        'axes.axisbelow': True           # 网格线在数据下方（<a target="_blank" href="https://www.osgeo.cn/matplotlib/users/dflt_style_changes.html" class="hitref" data-title="更改为默认样式— Matplotlib 3.3.3 文档" data-snippet='记号和网格现在绘制在实体元素（如填充轮廓）的上方，但绘制在线的下方。若要返回到上一个绘制线上方刻度和网格的行为，请设置 rcParams[&#39;axes.axisbelow&#39;] = False .' data-url="https://www.osgeo.cn/matplotlib/users/dflt_style_changes.html">6</a>）
    })

    # 保持seaborn主题但覆盖网格颜色（可选）
    sns.set_theme(style="whitegrid", rc={'grid.color': '#e0e0e0', 'grid.alpha': 0.3})

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlabel('time [s]', fontsize=12, labelpad=0)
    ax.set_ylabel('solving time [ms]', fontsize=12, labelpad=0)
    ax.tick_params(axis='both', labelsize=10, colors='#444444')
    plt.tight_layout()



    # 动态障碍
    cbf_2 = []
    min_d2 = 99
    t2 = 0
    for id in range(ob_num_max):
        h = []
        # for x in self.mpc.data['_x']:
        for j in range(robot_size2):
            obs = (ob[f'x{id}'][j], ob[f'y{id}'][j], ob[f'r{id}'][j])
            s = [robot_2['x'][j], robot_2['y'][j]]
            h.append(cbf_h(s, obs, robot_r))
            d2 = sqrt((s[0] - obs[0])**2 + (s[1] - obs[1])**2) - (robot_r + obs[2])
            if d2 < min_d2:
                min_d2 = d2
                t2 = j * 0.1
        cbf_2.append(h)
    

    cbf_5 = []
    min_d5 = 99
    t5 = 0
    for id in range(ob_num_max):
        h = []
        # for x in self.mpc.data['_x']:
        for j in range(robot_size5):
            obs = (ob[f'x{id}'][j], ob[f'y{id}'][j], ob[f'r{id}'][j])
            s = [robot_5['x'][j], robot_5['y'][j]]
            h.append(cbf_h(s, obs, robot_r))
            d5 = sqrt((s[0] - obs[0])**2 + (s[1] - obs[1])**2) - (robot_r + obs[2])
            if d5 < min_d5:
                min_d5 = d5
                t5 = j * 0.1

        cbf_5.append(h)   


    # plt.xlim(-0.2, T_2[-1]+0.2)
    # plt.ylim(-0.2, 12.2)
    plt.xlim(4.2, 6.2)
    plt.ylim(-0.2, 1.2)
    # ax.set_aspect('equal', adjustable='box')  # 保持比例同时锁定坐标范围 

    for i in range(len(cbf_2)):
        ax.plot(T_2[:], cbf_2[:][i], color='r', label="N-DCBF")
    for i in range(len(cbf_5)):
        ax.plot(T_5[:], cbf_5[:][i], color='c', label="C-RCBF\n(proposed)")    

    # plt.axvline(x=t2, color='r', linestyle='--')
    plt.axvline(x=t5, color='#6C3483', linestyle='--')
        
    plt.axhline(y=0, color='k', linestyle='--')
    plt.axhline(y=safety_dist, color='#DC7633', linestyle='--')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('Effective distance between AMR and obstacles [m]')

    ax.set_facecolor('white')  # 设置坐标轴背景为白色
    plt.tight_layout()
    # 创建第一个图例（原有元素，右下角）    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  
    original_legend = ax.legend(
        handles=by_label.values(),
        labels=by_label.keys(),
        loc='upper left',
        framealpha=0.9,
        title_fontsize=10,
        borderpad=0.3         # 图例内边距
    )

    ax.add_artist(original_legend)  # 必须保留原有图例对象

    # Save the figure as an image
    plt.savefig(results_rootpath + '/co_dist.png')
    plt.close()


''' @brief: Plotting the control input (linear and angular velocities)'''
def controlPlotting(results_rootpath):

    # 读取CSV文件
    robot_2_ = pd.read_csv(results_rootpath + '/N-DCBF.csv')   
    robot_5_ = pd.read_csv(results_rootpath + '/COAA-RCBF.csv')  

    robot_size2 = len(robot_2_)

    robot_size5 = len(robot_5_)
    if robot_size2 >= robot_size5:
        size = robot_size2
    else:
        size = robot_size5
    # 如果数据长度大于 max_num，适当跳过一些数据
    max_num = 300
    if size > max_num:
        robot_indices = np.linspace(0, size - 1, max_num).astype(int)
        robot_2 = robot_2_.iloc[robot_indices].reset_index(drop=True)  # 重采样后的障碍物数据
        robot_5 = robot_5_.iloc[robot_indices].reset_index(drop=True)  # 重采样后的障碍物数据
        size = max_num
    else:
        robot_2 = robot_2_
        robot_5 = robot_5_

    # Ensure T_2 matches the length of robot_2['t_mppi']
    T_2 = np.arange(0, 0.1 * robot_size2, 0.1)[:len(robot_2['t_mppi'])]
    T_5 = np.arange(0, 0.1 * robot_size5, 0.1)[:len(robot_5['t_mppi'])]

    # 全局设置（背景保持白色，网格参数调整）
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.grid': True,               # 开启网格线（<a target="_blank" href="https://www.cnblogs.com/ivanlee717/p/17483909.html" class="hitref" data-title="matplot画网格线分层级的用法- ivanlee717 - 博客园" data-snippet='Matplotlib 的 grid() 函数可以用于绘制网格线。 该函数的常用参数如下： - b 设置是否显示网格线。 可以取值为True 或False。' data-url="https://www.cnblogs.com/ivanlee717/p/17483909.html">1</a><a target="_blank" href="http://www.runoob.com/matplotlib/matplotlib-grid.html" class="hitref" data-title="Matplotlib 网格线 - 菜鸟教程" data-snippet='Matplotlib 网格线我们可以使用pyplot 中的grid() 方法来设置图表中的网格线。 grid() 方法语法格式如下： matplotlib.pyplot.grid(b=None, which=&#39;major&#39;, axis=&#39;both&#39; ...' data-url="http://www.runoob.com/matplotlib/matplotlib-grid.html">7</a>）
        'grid.color': '#e0e0e0',         # 浅灰色（HEX格式，<a target="_blank" href="https://wenku.csdn.net/answer/63f85veryr" class="hitref" data-title="matplotlib 浅灰色 - CSDN文库" data-snippet='要设置matplotlib绘图的浅灰色，可以使用RGB颜色代码表示。以下是一个示例代码，用于设置浅灰色背景： import matplotlib.pyplot as plt # 设置浅灰色RGB ...' data-url="https://wenku.csdn.net/answer/63f85veryr">4</a><a target="_blank" href="http://www.runoob.com/matplotlib/matplotlib-grid.html" class="hitref" data-title="Matplotlib 网格线 - 菜鸟教程" data-snippet='Matplotlib 网格线我们可以使用pyplot 中的grid() 方法来设置图表中的网格线。 grid() 方法语法格式如下： matplotlib.pyplot.grid(b=None, which=&#39;major&#39;, axis=&#39;both&#39; ...' data-url="http://www.runoob.com/matplotlib/matplotlib-grid.html">7</a>）
        'grid.alpha': 0.3,               # 透明度（<a target="_blank" href="https://www.python91.com/python_Matplotlib/187.html" class="hitref" data-title="matplotlib中的grid()方法如何设置网格线外观 - python编程" data-snippet='设置透明度：可以使用alpha参数来设置网格线的透明度，例如grid(alpha=0.5)可以将网格线的透明度设置为0.5。 综合使用这些参数可以实现各种网格线的外观设置 ...' data-url="https://www.python91.com/python_Matplotlib/187.html">10</a>）
        'grid.linestyle': '--',          # 虚线样式（可选）
        'axes.axisbelow': True           # 网格线在数据下方（<a target="_blank" href="https://www.osgeo.cn/matplotlib/users/dflt_style_changes.html" class="hitref" data-title="更改为默认样式— Matplotlib 3.3.3 文档" data-snippet='记号和网格现在绘制在实体元素（如填充轮廓）的上方，但绘制在线的下方。若要返回到上一个绘制线上方刻度和网格的行为，请设置 rcParams[&#39;axes.axisbelow&#39;] = False .' data-url="https://www.osgeo.cn/matplotlib/users/dflt_style_changes.html">6</a>）
    })

    # 保持seaborn主题但覆盖网格颜色（可选）
    sns.set_theme(style="whitegrid", rc={'grid.color': '#e0e0e0', 'grid.alpha': 0.3})

    plt.figure()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))  # 2行1列
    fig.subplots_adjust(hspace=0.4)

    ax1.plot(T_2[:], robot_2['v'][:], color='r', label="N-DCBF")
    ax1.plot(T_5[:], robot_5['v'][:], color='c', label="C-RCBF\n(proposed)")

    ax1.set_xlabel('time [s]', fontsize=12, labelpad=0)
    ax1.set_ylabel('Linear velocity [m/s]', fontsize=12, labelpad=0)
    ax1.tick_params(axis='both', labelsize=10, colors='#444444')
    plt.tight_layout()


    ax1.set_xlim(-0.2, T_2[-1]+0.2)
    ax1.set_ylim(-0.2, 2.2)
    # ax1.set_aspect('equal', adjustable='box')  # 保持比例同时锁定坐标范围

    # 创建第一个图例（原有元素，右下角）    
    handles1, labels1 = ax1.get_legend_handles_labels()
    by_label1 = dict(zip(labels1, handles1))
    original_legend1 = ax1.legend(
        handles=by_label1.values(),
        labels=by_label1.keys(),
        loc='upper right',
        framealpha=0.9,
        title_fontsize=10,
        borderpad=0.3         # 图例内边距
    )

    ax1.add_artist(original_legend1)  # 必须保留原有图例对象
    ax1.set_facecolor('white')  # 设置坐标轴背景为白色


    ax2.plot(T_2[:], robot_2['w'][:], color='r', label="N-DCBF")
    ax2.plot(T_5[:], robot_5['w'][:], color='c', label="C-RCBF\n(proposed)")

    ax2.set_xlabel('time [s]', fontsize=12, labelpad=0)
    ax2.set_ylabel('Angular velocity [rad/s]', fontsize=12, labelpad=0)
    ax2.tick_params(axis='both', labelsize=10, colors='#444444')

    ax2.set_xlim(-0.2, T_2[-1]+0.2)
    ax2.set_ylim(-0.6, 0.6)
    # ax2.set_aspect('equal', adjustable='box')  # 保持比例同时锁定坐标范围

    # 创建第一个图例（原有元素，右下角）    
    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label2 = dict(zip(labels2, handles2))
    original_legend2 = ax2.legend(
        handles=by_label2.values(),
        labels=by_label2.keys(),
        loc='upper right',
        framealpha=0.9,
        title_fontsize=10,
        borderpad=0.3         # 图例内边距
    )

    ax2.add_artist(original_legend2)  # 必须保留原有图例对象
    ax2.set_facecolor('white')  # 设置坐标轴背景为白色

    save_path = results_rootpath + '/co_control2.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()    


if __name__ == "__main__":
    # 主函数用于创建并运行 mppiControllerNode 实例，依次调用以下步骤：
   results_rootpath = './co_result' 
   pathPlotting(1, 0.3, 0.6, [10.0, 0.0] ,results_rootpath)
   timePlotting(results_rootpath)
   plot_cbf(0.1, 1, 0.6, 0.3, results_rootpath)
   controlPlotting(results_rootpath)
