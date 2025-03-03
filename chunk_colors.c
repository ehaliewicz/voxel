#include "types.h"

typedef struct {
    u32 color_palette[4096];   // color palettes start at 1024 and double when necessary
    u16* color_buffer_ptrs[6]; // 256,128,64,32,16,8
    u16 palette_cap;
    u16 palette_len;
} chunk_colors;


void parse_chunk() {
    for(int y = 0; y < CHUNK_SIZE; y++) {
        for(int x = 0; x < CHUNK_SIZE; x++) {
            //parse_column();

        }
    } 
}

int find_col_in_palette(u32 color, u32* col_palette, u16 num_cols_in_palette) {
    for(int i = 0; i < num_cols_in_palette; i++) {
        if(col_palette[i] == color) { return i; }
    }
    return -1;
}

#define LOG2(X) ((unsigned) (8*sizeof (unsigned long) - __builtin_clzl((X)) - 1))

void parse_column(chunk_colors* chk_cols) {
    u32 column_buffer[256];

    for(int z = 0; z < 256; z++) {
        // add color
        u32 col = get_next_color();

        s32 pal_idx = find_col_in_palette(col);

        if(pal_idx == -1) {
            chk_cols->color_palette[chk_cols->palette_len++] = col;
        }

    }
    
    u32 color_buffer_ptr_idx = LOG2(num_colors)-3;
    if(chk_cols->color_buffer_ptrs[color_buffer_ptr_idx] == NULL) {
        // allocate bigger color buffer :(

        // copy all previous indexes AND their colors to this buffer :/
        

    }
    if(num_colors > 129) {
        
        if(chk_cols->color_buffer_ptrs[])
    }
    
}