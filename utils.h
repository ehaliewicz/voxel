#ifndef UTILS_H
#define UTILS_H

u16 u32abgr_to_u16(u32 color_val){
    u8 r = (color_val&0xFF)>>3;
    u8 g = ((color_val>>8)&0xFF)>>3;
    u8 b = ((color_val>>16)&0xFF)>>3;
    return (b<<10)|(g<<5)|r;
}

u32 u16_to_u32abgr(u16 color_val) {
    u8 r = (color_val & 0b11111);
    u8 g = (color_val >> 5) & 0b11111;
    u8 b = (color_val >> 10) & 0b11111;
    return (0xFF)<<24|(b<<19)|(g<<11)|(r<<3);
}

#endif