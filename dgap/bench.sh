#!/bin/bash

echo 'pmem: ' ${PMEM_PATH}
echo 'data: ' ${DATA_PATH}
make clean && make
echo 'PAGERANK '
for i in {1..5}
do
	./pr -B ${DATA_PATH}/orkut/output.base.el -D ${DATA_PATH}/orkut/output.dynamic.el -f ${PMEM_PATH}/orkut.pmem -r 1 -n 5 -a >> pr-results-pmdk.out
	rm ${PMEM_PATH}/orkut*
done
