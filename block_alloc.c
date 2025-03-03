#include "types.h"
#include "config.h"
// block allocator


typedef struct block_node block_node;

 // 32 block pointers per node
 // pointers only use 48 bits, so we have a bunch of bits at the top
 // we'll use four.  3 for size, and 1 for used/unused status

 #define BLOCKS_IN_NODE 8
struct block_node {
    u64 pointers[BLOCKS_IN_NODE];
    block_node *prev;
    block_node *next;
    u8 size:3;
};

/*
    block sizes at each level are
    128x128x1 => 16,384 bytes => x8 = 524288
    128x128x4 => 65,536 bytes
    128x128x16 => 262,144 bytes
    128x128x64 => 1,048,576 bytes
    128x128x256 => 4,194,304 bytes (should be extremely rare :) )

*/
block_node* block_lists[5] = {
    NULL, NULL, NULL, NULL, NULL
};

#define DIRTY_BIT_POS 59
#define USED_BIT_POS 60
#define SIZE_BIT_POS 61

#define DIRTY_MASK (0b1<<DIRTY_BIT_POS)
#define USED_MASK (0b1<<USED_BIT_POS)
#define SIZE_MASK (0b111<SIZE_BIT_POS)


u64 adorn_fresh_pointer(u8* ptr, int size) {
    u64 uptr = (u64)ptr;
    uptr &= ~DIRTY_MASK; // not dirty, we use calloc so all blocks are zero'd by default
    uptr &= ~USED_MASK; // clear used mask
    uptr &= ~SIZE_MASK; // clear size mask
    uptr |= (size << SIZE_BIT_POS);
    return uptr;
}

void* unadorn_pointer(u64 ptr) {
    u64 rptr = ptr;
    rptr &= (~DIRTY_MASK);
    rptr &= (~USED_MASK);
    rptr &= (~SIZE_MASK);
    return rptr;
}

u64 calc_block_size(int size) {
    u64 height = 1 << (size*2);
    u64 size_per_block = CHUNK_SIZE*CHUNK_SIZE*height;
    return size_per_block;
}

block_node* alloc_block_node(int size) {
    u64 block_size = calc_block_size(size);
    u64 total_size = block_size * BLOCKS_IN_NODE;
    u8* full_block = calloc(total_size, 1);

    block_node* blk_node_ptr = calloc(1, sizeof(block_node));
    blk_node_ptr->prev = NULL;
    blk_node_ptr->next = NULL;
    for(int i = 0; i < BLOCKS_IN_NODE; i++) {
        blk_node_ptr->pointers[i] = adorn_pointer(&full_block[(i*block_size)], size);
    }
    return blk_node_ptr;
}

void* fetch_free_block(int size) {
    block_node* ptr = block_lists[size];
    if(ptr == NULL) {
        ptr = alloc_block_node(size);
        block_lists[size] = ptr;
    }

    block_node* prev_ptr = NULL;
    u64 block_size = calc_block_size(size);

    while(1) {
        if(ptr == NULL) {
            ptr = alloc_block_node(size);
            if(prev_ptr != NULL) {
                prev_ptr->next = ptr;
                ptr->prev = prev_ptr;
            }
        }
        for(int i = 0; i < BLOCKS_IN_NODE; i++) {
            if((ptr->pointers[i] & USED_MASK) == 0) {
                // found an unused block :) 
                // clear if necessary, mark as dirty, and return
                if((ptr->pointers[i] & DIRTY_MASK)) {
                    memset(((void*)(ptr->pointers[i])), 0, block_size);
                }
                ptr->pointers[i] |= USED_MASK;
                ptr->pointers[i] |= DIRTY_MASK;
                return unadorn_pointer(ptr->pointers[i]);
            }
        }
        // couldn't find an unused block here, move on to the next
        prev_ptr = ptr;
        ptr = ptr->next;
    }
}

void release_block(u64* ptr) {
    // we have to loop through list..
}