def read_depthmap_to_1024(input, output, name):
    with open(input, "rb") as f:
        bytes = f.read()
        img_data = bytes[-(512*512*3):]
        img_data_filtered = [None]*(1024*1024)
        img_data = [pix for idx,pix in enumerate(img_data) if idx%3==0]
        
        cnt = 0

        for iy in range(512):
            uiy = (iy-1)%512
            diy = (iy+1)%512
            oy = iy*2
            for ix in range(512):
                lix = (ix-1)%512
                rix = (ix+1)%512
                ox = ix*2
                sample = img_data[iy*512+ix]

                up_left_sample = img_data[uiy*512+lix]
                up_sample = img_data[uiy*512+ix]
                up_right_sample = img_data[uiy*512+rix]

                left_sample = img_data[iy*512+lix]
                right_sample = img_data[iy*512+rix]

                down_left_sample = img_data[diy*512+lix]
                down_sample = img_data[diy*512+ix]
                down_right_sample = img_data[diy*512+rix]

                ul_out = int((up_left_sample+up_sample+left_sample+sample)/4.0)
                ur_out = int((up_sample+up_right_sample+sample+right_sample)/4.0)
                dl_out = int((left_sample+sample+down_left_sample+down_sample)/4.0)
                dr_out = int((sample+right_sample+down_sample+down_right_sample)/4.0)

                # output it four times to img data filter
                img_data_filtered[oy*1024+ox] = ul_out
                img_data_filtered[oy*1024+ox+1] = ur_out
                img_data_filtered[(oy+1)*1024+ox] = dl_out
                img_data_filtered[(oy+1)*1024+ox+1] = dr_out



        with open(output, "w") as of:
            of.write("#include <stdint.h>\nuint32_t {}[{}]  __attribute__ ((aligned (32)));\n".format(name, 1024*1024))
            of.write("uint32_t {}[{}] = ".format(name, 1024*1024) + "{")
            for idx,val in enumerate(img_data_filtered):
                if idx % 16 == 0:
                    of.write("\n")
                of.write("{}, ".format(val))
            of.write("};")

def read_ppm(dim, input, output, name, single_channel = False, double=False):
    outdim = dim*2 if double else dim
    with open(input, "rb") as f:
        bytes = f.read()
        img_data = bytes[-(dim*dim*3):]
        img_data_filtered = []
        
        if single_channel:
            cnt = 0

            for val in img_data:
                if cnt == 0:
                    img_data_filtered.append(val)
                cnt += 1
                if cnt == 3:
                    cnt = 0
        else:
            cnt = 0
            for val in img_data:
                    
                img_data_filtered.append(val)
                cnt += 1
                if cnt == 3:
                    img_data_filtered.append(255)
                    cnt = 0


        with open(output, "w") as of:
            of.write("#include <stdint.h>\nuint8_t {}[{}] = ".format(name, dim*dim* (1 if single_channel else 4)) + "{")
            for idx,val in enumerate(img_data_filtered):
                if idx % 16 == 0:
                    of.write("\n")
                of.write("{}, ".format(val))
            of.write("};")


if __name__ == '__main__':
    read_depthmap_to_1024("res/D10.ppm", "depth.c", "depthmap_u32s")
    #read_ppm(1024, "res/C10W.ppm", "color.c", "colormap")