# # def allocate_tiles_to_blocks(num_tiles, num_blocks):
# #     # 初始化每个 block 的 tile 列表
# #     block_allocation = [[] for _ in range(num_blocks)]

# #     # 分配 tile 到每个 block
# #     for tile_id in range(num_tiles):
# #         block_id = tile_id % num_blocks  # 计算当前 tile 应该分配到哪个 block
# #         block_allocation[block_id].append(tile_id)

# #     return block_allocation

# # # 示例参数
# # num_tiles = 24
# # num_blocks = 16

# # # 调用函数进行分配
# # allocation = allocate_tiles_to_blocks(num_tiles, num_blocks)

# # # 打印结果
# # for block_id, tiles in enumerate(allocation):
# #     print(f"Block {block_id} has tiles: {tiles}")

# # print(allocation)

def allocate_tiles_to_blocks(tile_ids, num_blocks):
    # 初始化每个 block 的 tile 列表
    block_allocation = [[] for _ in range(num_blocks)]
    tile_allocation = [[] for _ in range(len(tile_ids))]  # List of lists for tiles
    tile_per_block = len(tile_ids)/num_blocks #16block 24tile
    # 分配 tile 到每个 block
    if  tile_per_block > 1:
        for idx, tile_id in enumerate(tile_ids):
            block_id = idx % num_blocks  # 计算当前 tile 应分配到哪个 block
            block_allocation[block_id].append(tile_id)
            tile_allocation[idx].append(block_id)
    else:
        for i in range(num_blocks):
            # Assign tiles to blocks in a round-robin manner
            assigned_tile_idx = i % len(tile_ids)
            assigned_tile = tile_ids[assigned_tile_idx]
            block_allocation[i].append(assigned_tile)
            tile_allocation[assigned_tile_idx].append(i)
            
            # for tile_idx, blocks in enumerate(tile_allocation):
            #     print(f"Tile {tile_ids[tile_idx]}: {blocks}")
    
    return block_allocation

# 示例 tile IDs 和 block 数量
tile_ids = list(range(0, 24))  # [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
num_blocks = 16

# 调用分配函数
block_allocation = allocate_tiles_to_blocks(tile_ids, num_blocks)
print(block_allocation)
# 打印结果
# for block_id, tiles in enumerate(block_allocation):
#     print(f"Block {block_id}: {tiles}")


# def allocate_blocks_to_tiles(block_ids, num_tiles, blocks_per_tile):
#     """
#     Allocate blocks to tiles and vice versa.

#     :param block_ids: List of block IDs to be allocated.
#     :param num_tiles: Total number of tiles.
#     :param blocks_per_tile: Number of blocks each tile should handle.
#     :return: Two lists:
#         - block_allocation: A list where each element is a list of tile IDs for a given block.
#         - tile_allocation: A list where each element is a list of block IDs for a given tile.
#     """
#     # Initialize the allocation lists
#     block_allocation = [[] for _ in range(len(block_ids))]
#     tile_allocation = [[] for _ in range(num_tiles)]
    
#     # Total blocks
#     total_blocks = len(block_ids)

#     # Distribution of blocks to tiles
#     for i, block_id in enumerate(block_ids):
#         # Determine which tile this block will go to
#         tile_id = i % num_tiles
#         block_allocation[block_id].append(tile_id)
#         tile_allocation[tile_id].append(block_id)
    
#     # Handle excess blocks if the number of blocks is not a perfect multiple of num_tiles
#     if blocks_per_tile:
#         for tile_id in range(num_tiles):
#             # Calculate the starting and ending block index for each tile
#             start_idx = tile_id * blocks_per_tile
#             end_idx = min(start_idx + blocks_per_tile, total_blocks)
            
#             for block_idx in range(start_idx, end_idx):
#                 block_allocation[block_ids[block_idx]].append(tile_id)
#                 tile_allocation[tile_id].append(block_ids[block_idx])
    
#     return block_allocation, tile_allocation

# # Example usage
# block_ids = list(range(16))  # List of block IDs
# num_tiles = 24  # Total number of tiles
# blocks_per_tile = 16  # Number of blocks per tile (if applicable)

# # Call the allocation function
# block_allocation, tile_allocation = allocate_blocks_to_tiles(block_ids, num_tiles, blocks_per_tile)

# # Print results
# print("Block Allocation:")
# for block_id, tiles in enumerate(block_allocation):
#     print(f"Block {block_id}: {tiles}")

# print("\nTile Allocation:")
# for tile_id, blocks in enumerate(tile_allocation):
#     print(f"Tile {tile_id}: {blocks}")
