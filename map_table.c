
#ifndef MAP_TABLE_C
#define MAP_TABLE_C

#include <dirent.h> 
#include <stdlib.h>

int num_maps = 0;
int map_name_table_size = 0;
int map_name_table_cap = 0;
char* map_name_table = NULL;
int map_idxs[64];

void load_map_table() {
  if(map_name_table != NULL) { free(map_name_table); }
  DIR *d;
  struct dirent *dir;
  d = opendir("./maps");
  if (d) {
    while ((dir = readdir(d)) != NULL && num_maps < 64) {
      if(strstr(dir->d_name, ".vxl") == NULL) { continue; }
      while((map_name_table_size+dir->d_namlen+1) >= map_name_table_cap) {
        map_name_table_cap = (map_name_table_cap < 8 ? 8 : (map_name_table_cap * 1.5));
        map_name_table = realloc(map_name_table, sizeof(char*)*map_name_table_cap);
      }
      map_idxs[num_maps++] = map_name_table_size;
      strcpy(&map_name_table[map_name_table_size], dir->d_name);
      map_name_table_size += dir->d_namlen;
      map_name_table_size += 1;
    }
    closedir(d);
  }
}

#endif