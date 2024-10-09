# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import numpy as np
# import itertools
# from matplotlib.widgets import Slider

# def draw_pipeline(bank_num, tiles_id, bank_data):
#     layer_colors = {0: 'blue', 1: 'green', 2: 'red'}
#     sram_type_hatch = {'weight': '//', 'FI': '++', 'FO': 'oo'}

#     fig, ax = plt.subplots(figsize=(14, 8))
#     plt.subplots_adjust(left=0.1, right=0.85, bottom=0.2, top=0.9)

#     # Create slider axes and sliders
#     ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
#     slider = Slider(ax_slider, 'Zoom', 0.1, 10.0, valinit=1.0)

#     def update(val):
#         zoom = slider.val
#         ax.set_xlim(left=0, right=max(end_times) / zoom)
#         fig.canvas.draw_idle()

#     # Collect end times for setting x-axis limits
#     end_times = []
    
#     for (tile, bank), data in bank_data.items():
#         for i in range(len(data['start_time'])):
#             start = data['start_time'][i]
#             end = data['end_time'][i]
#             layer = data['layer_id'][i]
#             sram_type = data['sram_type'][i]
#             color = layer_colors[layer]
#             hatch = sram_type_hatch[sram_type]
#             if start > end:
#                 print(f"Tile {tile}, Bank {bank}: 开始时间 {start} 大于 结束时间 {end} (错误)")
#             if start == end:
#                 end += 1  # 添加一个小的时间增量，使其可见
#                 data['end_time'][i] = end

#             end_times.append(end)
#             ax.broken_barh([(start, end - start)], (tile * 10 + bank * 3, 2), facecolors=color, edgecolor='black', hatch=hatch)

#     # Setting labels and titles
#     ax.set_xlabel('Cycle')
#     ax.set_ylabel('Tile Bank')
#     ax.set_title('Pipeline Visualization')

#     # Creating a custom legend
#     layer_patches = [mpatches.Patch(color=layer_colors[i], label=f'Layer {i}') for i in layer_colors]
#     sram_patches = [mpatches.Patch(facecolor='white', edgecolor='black', hatch=sram_type_hatch[key], label=key) for key in sram_type_hatch]
#     ax.legend(handles=layer_patches + sram_patches, loc='upper left', bbox_to_anchor=(1, 1))

#     # Setting Y-axis ticks
#     flattened_tiles_id = list(itertools.chain.from_iterable(tiles_id))
#     max_value = max(flattened_tiles_id)
#     ax.set_yticks([tile * 10 + bank * 3 + 1 for tile in range(max_value + 1) for bank in range(bank_num)])
#     ax.set_yticklabels([f'Tile {tile}, Bank {bank}' for tile in range(max_value + 1) for bank in range(bank_num)])

#     # Initialize slider
#     update(slider.val)
#     plt.show()
#     output_filename = 'draft.png'
#     plt.savefig(output_filename)

#     # 关闭图表以释放内存
#     plt.close()
#     print(f"图表已保存为 {output_filename}")

# # 示例调用（你需要替换为实际数据）
# tiles_id = [[0,1],[1],[0,1,2]]
# bank_data = {
#     (0, 0): {'layer_id': [0, 0, 2], 'sram_type': ['weight', 'FI', 'FO'], 'start_time': [0, 16, 104], 'end_time': [16, 19, 212]},
#     (0, 1): {'layer_id': [0, 1, 2], 'sram_type': ['weight', 'FO', 'FI'], 'start_time': [0, 425334, 1594975], 'end_time': [212711, 638127, 1595005]},
#     # Add more tile-bank data here...
# }

# draw_pipeline(bank_num=3, tiles_id=tiles_id, bank_data=bank_data)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import itertools
# Example data structure (adjust with your actual data):
# tiles_id = [[0,1],[1],[0,1,2]]
# bank_data = {
#     (0, 0): {'layer_id': [0, 0, 2], 'sram_type': ['weight', 'FI', 'FO'], 'start_time': [0, 16, 104], 'end_time': [16, 19, 212]},
#     (0, 1): {'layer_id': [0, 1, 2], 'sram_type': ['weight', 'FO', 'FI'], 'start_time': [0, 425334, 1594975], 'end_time': [212711, 638127, 1595005]},
#     # Add more tile-bank data here...
# }

def draw_pipeline(bank_num,tiles_id,bank_data,Compute_latency_ex=None,Noc=None):
    layer_colors = {0: 'blue', 1: 'green', 2: 'red'}
    sram_type_hatch = {'weight': '//', 'FI': '++', 'FO': 'oo'}
    fig, ax = plt.subplots(figsize=(14, 8))
    scarce = 0
    for (tile, bank), data in bank_data.items():
        print(f"scarce:{scarce}")
        count = 0
        scarce = 0
        
        #print(f"tile{tile},bank{bank},num:{len(data['start_time'])}")
        for i in range(len(data['start_time'])):
            start = data['start_time'][i]
            end = data['end_time'][i]
            layer = data['layer_id'][i]
            sram_type = data['sram_type'][i]
            color = layer_colors[layer]
            hatch = sram_type_hatch[sram_type]
            
            if start > end:
                print(f"Tile {tile}, Bank {bank}: 开始时间 {start} 大于 结束时间 {end} (错误)")
                print(f"tile{tile},bank{bank},num_ideal:{len(data['start_time'])},num_actual:{count}")

                continue
            if start == end:
                end += 1  # 添加一个小的时间增量，使其可见
                data['end_time'][i] = end
                scarce = scarce + 1

            count = count + 1
            ax.broken_barh([(start, end - start)], (tile * 10 + bank * 3, 2), facecolors=color, edgecolor='black', hatch=hatch)
    for tile_id in range(len(Compute_latency_ex)):
        for layer_id in range(len(Compute_latency_ex[tile_id].Compute_start_time)):
            #print(f"Tile{tile_id},Layer{Layer_id} : Start Time{Compute_latency_ex[tile_id].Compute_start_time[Layer_id]}, Duration{Compute_latency_ex[tile_id].Compute_duration[Layer_id]}, End Time{Compute_latency_ex[tile_id].Compute_end_time[Layer_id]}")
            start = Compute_latency_ex[tile_id].Compute_start_time[layer_id]
            end = Compute_latency_ex[tile_id].Compute_end_time[layer_id]
            color = layer_colors[Compute_latency_ex[tile_id].layer_id[layer_id]]
            hatch = 'x'
            ax.broken_barh([(start, end - start)], (3 * 10 + tile_id * 3, 2), facecolors=color, edgecolor='black', hatch=hatch)
    for layer_id in range(len(Noc.Noc_start_time)):
        start = Noc.Noc_start_time[layer_id]
        end = Noc.Noc_end_time[layer_id]
        color = layer_colors[layer_id]
        hatch = 'x'
        ax.broken_barh([(start, end - start)], (4 * 10 + 0 * 3, 2), facecolors=color, edgecolor='black', hatch=hatch)
    
    # Setting labels and titles
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Tile Bank')
    ax.set_title('Pipeline Visualization')
    #ax.set_xscale('log')

    # Adjust x-axis ticks for log scale
    #ax.get_xaxis().set_major_formatter(ScalarFormatter())

    #ax.set_xlim(0, 60)  # 根据数据的实际范围调整
    #ax.set_ylim(0, 50)       # 根据条形的位置调整

    # Creating a custom legend
    layer_patches = [mpatches.Patch(color=layer_colors[i], label=f'Layer {i}') for i in layer_colors]
    sram_patches = [mpatches.Patch(facecolor='white', edgecolor='black', hatch=sram_type_hatch[key], label=key) for key in sram_type_hatch]
    ax.legend(handles=layer_patches + sram_patches, loc='upper left', bbox_to_anchor=(1, 1))

    # Setting Y-axis ticks
    flattened_tiles_id = list(itertools.chain.from_iterable(tiles_id))
    # 找到最大值
    max_value = max(flattened_tiles_id)
    b=[tile * 10 + bank * 3 + 1 for tile in range(max_value+1) for bank in range(bank_num)]
    b.extend([3 * 10 + 1,3*10+1*3+1,10*3+2*3+1,10*3+3*3+1])
    
    ax.set_yticks(b)
    a=[f'Tile {tile}, Bank {bank}' for tile in range(max_value+1) for bank in range(bank_num)]
    a.extend(['Tile 1','Tile 2','Tile 3','Noc'])
    ax.set_yticklabels(a)

    output_filename = 'draft2.png'#bs stage
    plt.savefig(output_filename)

    # 关闭图表以释放内存
    plt.close()
    print(count)
    print(f"图表已保存为 {output_filename}")
