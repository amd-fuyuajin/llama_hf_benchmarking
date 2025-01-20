#!/bin/bash

OUTPUT_DIR=$1

uname -a > "$OUTPUT_DIR"/uname.txt
lscpu > "$OUTPUT_DIR/cpu_info.txt"
lshw -c memory > "$OUTPUT_DIR/memory_info.txt"
numactl -H > "$OUTPUT_DIR/numa_nodes.txt"
dmidecode -t 17 > "$OUTPUT_DIR/dmidecode_memory.txt"
dmidecode -t bios > "$OUTPUT_DIR/bios_info.txt"
