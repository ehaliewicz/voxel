import numpy as np
from utils import color_tuple_to_int, clampf
# code to load voxlap map files
import random




# let's just do up to 32 spans
# and up to 128 colors, per column
def read_voxlap_column(bytes, byte_idx, spans, colors):
   
    z = 0
    color_idx = 0
    span_idx = 0
    while True:
        i = z
        num_4byte_chunks = bytes[byte_idx]
        top_color_start = bytes[byte_idx+1]
        top_color_end = bytes[byte_idx+2]
        bot_color_start = None
        bot_color_end = None
        len_top = None
        len_bot = None

        len_bot = top_color_end - top_color_start + 1
        src_colors_idx = byte_idx+4
        for i in range(top_color_start, top_color_end+1):
            col = (bytes[src_colors_idx+2],bytes[src_colors_idx+1],bytes[src_colors_idx+0],bytes[src_colors_idx+3])
            colors[color_idx] = color_tuple_to_int(col)
            color_idx += 1
            src_colors_idx += 4
        
        if num_4byte_chunks == 0:
            # end of column
            byte_idx += 4 * (len_bot + 1)
            spans[span_idx][0] = max(1, 63-top_color_start)
            spans[span_idx][1] = max(0, 63-(top_color_end+1))
            spans[span_idx][2] = 0 #63
            spans[span_idx][3] = 0 #63
            span_idx += 1
            break
        
        len_top = (num_4byte_chunks-1) - len_bot
        byte_idx += bytes[byte_idx]*4
        bot_color_end = bytes[byte_idx+3]
        bot_color_start = bot_color_end - len_top
        for i in range(bot_color_start, bot_color_end):
            factor = bytes[src_colors_idx+3]/255.0
            col = (int(bytes[src_colors_idx+2]*factor)<<16 |
                   int(bytes[src_colors_idx+1]*factor)<<8 |
                   int(bytes[src_colors_idx+0]*factor)) #   2,0,1,  2,1,0
                   #bytes[src_colors_idx+3])
            colors[color_idx] = col#color_tuple_to_int( col )
            color_idx += 1
            src_colors_idx += 4

        spans[span_idx][0] = max(1, 63-top_color_start)
        spans[span_idx][1] = 63-(top_color_end+1)
        spans[span_idx][2] = 63-bot_color_start
        spans[span_idx][3] = 63-bot_color_end
        span_idx += 1
        
    return span_idx, color_idx, byte_idx
         

    # span is length_of_span_data*4 bytes


def load_voxlap_map(f, colors_data, spans_data, columns_data):

    with open(f, "rb") as f:
        all_bytes = f.read(-1)
        byte_idx = 0
        for y in range(512):
            for x in range(512):
                span_idx, color_idx, byte_idx = read_voxlap_column(all_bytes, byte_idx, spans_data[y][x], colors_data[y][x])
                columns_data[y][x][0] = spans_data[y][x][0][0]
                columns_data[y][x][1] = span_idx
        return spans_data, colors_data, columns_data
