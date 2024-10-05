import matplotlib.pyplot as plt
import math
from hardware import Bank,Buffer,MFU,NoC_Router,Hardware,Tile

# 所有time的情况
def draw(hardware:Hardware,fusion_group_id,tile_id):
    # 提取数据
    tile = hardware.Tile_groups[fusion_group_id][tile_id] 
    operations = {
        'HBM Load': list(zip(hardware.HBM.start_time_list, 
                            [end - start for start, end in zip(hardware.HBM.start_time_list, 
                                                            hardware.HBM.end_time_list)])),
        'Bank Param 1': list(zip(tile.Buffer.Param_bank[0][0].start_time_list, 
                                    [end - start for start, end in zip(tile.Buffer.Param_bank[0][0].start_time_list, 
                                                                        tile.Buffer.Param_bank[0][0].end_time_list)])),
        'Bank Param 2': list(zip(tile.Buffer.Param_bank[1][0].start_time_list, 
                                    [end - start for start, end in zip(tile.Buffer.Param_bank[1][0].start_time_list, 
                                                                        tile.Buffer.Param_bank[1][0].end_time_list)])),
        'NoC Communication': list(zip(hardware.NoC_router.start_time_list, 
                            [end - start for start, end in zip(hardware.NoC_router.start_time_list, 
                                                            hardware.NoC_router.end_time_list)])),
        'Bank FI 1': list(zip(tile.Buffer.FI_bank[0][0].start_time_list, 
                                    [end - start for start, end in zip(tile.Buffer.FI_bank[0][0].start_time_list, 
                                                                    tile.Buffer.FI_bank[0][0].end_time_list)])),
        'Bank FI 2': list(zip(tile.Buffer.FI_bank[1][0].start_time_list, 
                                    [end - start for start, end in zip(tile.Buffer.FI_bank[1][0].start_time_list, 
                                                                    tile.Buffer.FI_bank[1][0].end_time_list)])),
        'Compute': list(zip(tile.MFU.start_time_list, 
                            [end - start for start, end in zip(tile.MFU.start_time_list, 
                                                            tile.MFU.end_time_list)])),
        'Bank FO': list(zip(tile.Buffer.FO_bank[0][0].start_time_list, 
                                [end - start for start, end in zip(tile.Buffer.FO_bank[0][0].start_time_list, 
                                                                    tile.Buffer.FO_bank[0][0].end_time_list)])),
    }
    
    
    operations = dict(reversed(list(operations.items())))
    # 为绘图分配不同的颜色
    colors = {
        'Bank FI 1': 'skyblue',
        'Bank FI 2': 'blue',
        'Bank FO': 'lightgreen',
        'Bank Param 1': 'orange',
        'Bank Param 2': 'red',
        'Compute': 'purple',
        'HBM Load': 'yellow',
        'NoC Communication': 'pink'
    }

    # 分配每个操作类型在 Y 轴的位置
    y_labels = list(operations.keys())
    y_pos = range(len(y_labels))

    # 绘制甘特图
    fig, ax = plt.subplots(figsize=(10, 4))

    for i, label in enumerate(y_labels):
        for (start, duration) in operations[label]:
            ax.broken_barh([(start, duration)], (i - 0.5, 0.5), facecolors=colors.get(label, 'grey'))

    #ax.set_ylim(-1, len(y_labels))
    ax.set_ylim(-0.5, len(y_labels) - 0.5)  # 缩小y轴范围，去掉多余的空白
    min_start_time = min([start for sublist in operations.values() for (start, duration) in sublist])
    max_end_time = max([end for sublist in operations.values() for (start, duration) in sublist for end in [start + duration]])

    # 动态设置 x 轴范围
    ax.set_xlim(min_start_time, max_end_time)
    ax.autoscale(enable=False)
    #ax.set_xlim(0, max([end for sublist in operations.values() for (start, duration) in sublist for end in [start + duration]]))
    ax.set_xlabel('Time')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.grid(True)

    plt.title('Tile Components Operations Timeline')
    plt.savefig('tile_operations_timeline_total.png', bbox_inches='tight')

    # 关闭图形以释放内存
    plt.close()

# 用于放大三个小块计算访存过程细看一下流程
def draw_sample(hardware:Hardware,fusion_group_id,tile_id):
    # 提取数据
    tile = hardware.Tile_groups[fusion_group_id][tile_id] 
    
    # operations = {
        
    #     'HBM Load': list(zip(hardware.HBM.start_time_list[:6], 
    #                         [end - start for start, end in zip(hardware.HBM.start_time_list[:6], 
    #                                                         hardware.HBM.end_time_list[:6])])),
    #     'Bank Param 1': list(zip(tile.Buffer.Param_bank[0][0].start_time_list[:6], 
    #                                 [end - start for start, end in zip(tile.Buffer.Param_bank[0][0].start_time_list[:6], 
    #                                                                     tile.Buffer.Param_bank[0][0].end_time_list[:6])])),
    #     'Bank Param 2': list(zip(tile.Buffer.Param_bank[1][0].start_time_list[:6], 
    #                                 [end - start for start, end in zip(tile.Buffer.Param_bank[1][0].start_time_list[:6], 
    #                                                                    tile.Buffer.Param_bank[1][0].end_time_list[:6])])),
    #     'NoC Communication': list(zip([hardware.NoC_router.start_time_list[0]], 
    #                         [end - start for start, end in zip([hardware.NoC_router.start_time_list[0]], 
    #                                                         [hardware.NoC_router.end_time_list[0]])])),
    #     'Bank FI 1': list(zip(tile.Buffer.FI_bank[0][0].start_time_list[:3], 
    #                                 [end - start for start, end in zip(tile.Buffer.FI_bank[0][0].start_time_list[:3], 
    #                                                                 tile.Buffer.FI_bank[0][0].end_time_list[:3])])),
    #     'Bank FI 2': list(zip(tile.Buffer.FI_bank[1][0].start_time_list[:3], 
    #                                 [end - start for start, end in zip(tile.Buffer.FI_bank[1][0].start_time_list[:3], 
    #                                                                 tile.Buffer.FI_bank[1][0].end_time_list[:3])])),
    #     'Compute': list(zip(tile.MFU.start_time_list[:12], 
    #                         [end - start for start, end in zip(tile.MFU.start_time_list[:12], 
    #                                                         tile.MFU.end_time_list[:12])])),
    #     'Bank FO': list(zip(tile.Buffer.FO_bank[0][0].start_time_list[:12], 
    #                             [end - start for start, end in zip(tile.Buffer.FO_bank[0][0].start_time_list[:12], 
    #                                                                 tile.Buffer.FO_bank[0][0].end_time_list[:12])])),
    # }
    operations = {
        # 'HBM Load': list(zip(hardware.HBM.start_time_list[-12:], 
        #                     [end - start for start, end in zip(hardware.HBM.start_time_list[-12:], 
        #                                                     hardware.HBM.end_time_list[-12:])])),
        'Bank Param 1': list(zip(tile.Buffer.Param_bank[0][0].start_time_list[-12:], 
                                    [end - start for start, end in zip(tile.Buffer.Param_bank[0][0].start_time_list[-12:], 
                                                                        tile.Buffer.Param_bank[0][0].end_time_list[-12:])])),
        'Bank Param 2': list(zip(tile.Buffer.Param_bank[1][0].start_time_list[-12:], 
                                    [end - start for start, end in zip(tile.Buffer.Param_bank[1][0].start_time_list[-12:], 
                                                                        tile.Buffer.Param_bank[1][0].end_time_list[-12:])])),
        'NoC Communication': list(zip([hardware.NoC_router.start_time_list[0]], 
                            [end - start for start, end in zip([hardware.NoC_router.start_time_list[0]], 
                                                            [hardware.NoC_router.end_time_list[0]])])),
        'Bank FI 1': list(zip(tile.Buffer.FI_bank[0][0].start_time_list[-12:], 
                                    [end - start for start, end in zip(tile.Buffer.FI_bank[0][0].start_time_list[-12:], 
                                                                    tile.Buffer.FI_bank[0][0].end_time_list[-12:])])),
        'Bank FI 2': list(zip(tile.Buffer.FI_bank[1][0].start_time_list[-12:], 
                                    [end - start for start, end in zip(tile.Buffer.FI_bank[1][0].start_time_list[-12:], 
                                                                    tile.Buffer.FI_bank[1][0].end_time_list[-12:])])),
        'Compute': list(zip(tile.MFU.start_time_list[-12:], 
                            [end - start for start, end in zip(tile.MFU.start_time_list[-12:], 
                                                            tile.MFU.end_time_list[-12:])])),
        'Bank FO': list(zip(tile.Buffer.FO_bank[0][0].start_time_list[-12:], 
                                [end - start for start, end in zip(tile.Buffer.FO_bank[0][0].start_time_list[-12:], 
                                                                    tile.Buffer.FO_bank[0][0].end_time_list[-12:])])),
    }
    operations = dict(reversed(list(operations.items())))
    # 为绘图分配不同的颜色
    colors = {
        'Bank FI 1': 'skyblue',
        'Bank FI 2': 'blue',
        'Bank FO': 'lightgreen',
        'Bank Param 1': 'orange',
        'Bank Param 2': 'red',
        'Compute': 'purple',
        # 'HBM Load': 'yellow',
        'NoC Communication': 'pink'
    }

    # 分配每个操作类型在 Y 轴的位置
    y_labels = list(operations.keys())
    y_pos = range(len(y_labels))

    # 绘制甘特图
    fig, ax = plt.subplots(figsize=(16, 8))

    for i, label in enumerate(y_labels):
        for (start, duration) in operations[label]:
            ax.broken_barh([(start, duration)], (i - 0.5, 1), facecolors=colors.get(label, 'grey'),edgecolors='black')

    #ax.set_ylim(-1, len(y_labels))
    ax.set_ylim(-0.5, len(y_labels) - 0.5)  # 缩小y轴范围，去掉多余的空白
    # 找到最早的 start_time 和最晚的 end_time
    min_start_time = min([start for sublist in operations.values() for (start, duration) in sublist])
    max_end_time = max([end for sublist in operations.values() for (start, duration) in sublist for end in [start + duration]])

    # 动态设置 x 轴范围
    ax.set_xlim(min_start_time, max_end_time)
    ax.autoscale(enable=False)
    #ax.set_xlim(0, max([end for sublist in operations.values() for (start, duration) in sublist for end in [start + duration]]))
    ax.set_xlabel('Time')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.grid(True)

    plt.title('Tile Components Operations Timeline')
    plt.savefig('tile_operations_timeline.png', bbox_inches='tight')

    # 如果需要保存为其他格式，例如 PDF：
    # plt.savefig('tile_operations_timeline.pdf', bbox_inches='tight')

    # 关闭图形以释放内存
    plt.close()