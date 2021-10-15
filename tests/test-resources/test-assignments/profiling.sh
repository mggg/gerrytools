
subdir="./test-resources/test-assignments/"
gprof2dot -f pstats ${subdir}compress.pstats | dot -Tpng -o ${subdir}compress.png
gprof2dot -f pstats ${subdir}decompress.pstats | dot -Tpng -o ${subdir}decompress.png
