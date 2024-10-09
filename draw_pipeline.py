import matplotlib.pyplot as plt
import math
from hardware import Bank,Buffer,MFU,NoC_Router,Hardware,Tile
def extract_time_info(hardware, tile, keys_to_extract, sources):
    time_data = {
        'start_time': {},
        'end_time': {}
    }
    
    for source in sources:
        # 确定数据源是 hardware 还是 tile，并提取相应的数据
        # if source.startswith('tile.'):
        #     # 对于 tile 属性
        #     source_obj = getattr(tile, source.split('.')[1])
        if source == 'tile.Buffer.Param_bank[0][0]':
            source_obj = tile.Buffer.Param_bank[0][0]
        elif source == 'tile.Buffer.Param_bank[1][0]':
            source_obj = tile.Buffer.Param_bank[1][0]
        elif source == 'tile.Buffer.FI_bank[0][0]':
            source_obj = tile.Buffer.FI_bank[0][0]
        elif source == 'tile.Buffer.FI_bank[1][0]':
            source_obj = tile.Buffer.FI_bank[1][0]
        elif source == 'tile.Buffer.FO_bank[0][0]':
            source_obj = tile.Buffer.FO_bank[0][0]
        elif source == 'tile.MFU':
            source_obj = tile.MFU
        else:
            # 对于 hardware 属性
            source_obj = getattr(hardware, source)

        # 提取 start_time 和 end_time
        start_times = []
        end_times = []
        
        for key in keys_to_extract:
            if key in source_obj.time_stamp['start_time']:
                start_times.extend(source_obj.time_stamp['start_time'][key])
                end_times.extend(source_obj.time_stamp['end_time'][key])
        
        # 将提取的时间信息存入字典
        time_data['start_time'][source] = start_times
        time_data['end_time'][source] = end_times
    
    return time_data
def draw_time_stamp(hardware:Hardware,fusion_group_id,tile_id):
    # 提取数据
    tile = hardware.Tile_groups[fusion_group_id][tile_id] # nocend:612095 对上了
    keys_to_extract = list(range(623874,624383))#512个c#623，872；97795, 98292; 110,597 ;614,400;7组是正常的，因为hbm的原因，7组没有连起来；同时hbm因为noc会断开
    source = ['HBM','tile.Buffer.Param_bank[0][0]','tile.Buffer.Param_bank[1][0]','NoC_router','tile.Buffer.FI_bank[0][0]','tile.Buffer.FI_bank[1][0]','tile.MFU','tile.Buffer.FO_bank[0][0]'] 
    time_data = extract_time_info(hardware,tile,keys_to_extract,source)
    print(1)
    operations = {
        # 'HBM Load': list(zip(time_data['start_time'][source[0]], 
        #                     [end - start for start, end in zip(time_data['start_time'][source[0]], 
        #                                                     time_data['end_time'][source[0]])])),
        'Bank Param 1': list(zip(time_data['start_time'][source[1]], 
                                    [end - start for start, end in zip(time_data['start_time'][source[1]], 
                                                                        time_data['end_time'][source[1]])])),
        'Bank Param 2': list(zip(time_data['start_time'][source[2]], 
                                    [end - start for start, end in zip(time_data['start_time'][source[2]], 
                                                                        time_data['end_time'][source[2]])])),
        'NoC Communication': list(zip(time_data['start_time'][source[3]], 
                            [end - start for start, end in zip(time_data['start_time'][source[3]], 
                                                            time_data['end_time'][source[3]])])),
        'Bank FI 1': list(zip(time_data['start_time'][source[4]], 
                                    [end - start for start, end in zip(time_data['start_time'][source[4]], 
                                                                    time_data['end_time'][source[4]])])),
        'Bank FI 2': list(zip(time_data['start_time'][source[5]], 
                                    [end - start for start, end in zip(time_data['start_time'][source[5]], 
                                                                    time_data['end_time'][source[5]])])),
        'Compute': list(zip(time_data['start_time'][source[6]], 
                            [end - start for start, end in zip(time_data['start_time'][source[6]], 
                                                            time_data['end_time'][source[6]])])),
        'Bank FO': list(zip(time_data['start_time'][source[7]], 
                                [end - start for start, end in zip(time_data['start_time'][source[7]], 
                                                                    time_data['end_time'][source[7]])])),
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
            ax.broken_barh([(start, duration)], (i - 0.5, 1), facecolors=colors.get(label, 'grey'), alpha=1.0)#,edgecolors = 'black')

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
    plt.savefig('tile_operations_timeline_total_try_attn.png', bbox_inches='tight',dpi=600)

    # 关闭图形以释放内存
    plt.close()

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