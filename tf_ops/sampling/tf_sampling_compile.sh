#/bin/bash
# /usr/local/cuda-10.1/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

# g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC \
#   -I ${TF_INC} \
#   -I ${TF_INC}/external/nsync/public \
#   -I /usr/local/cuda-10.1/include -lcudart -L /usr/local/cuda-10.1/lib64/ \
#   -L${TF_LIB} -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

TF_CFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CUDA_ROOT=/usr/local/cuda

${CUDA_ROOT}/bin/nvcc \
./tf_sampling_g.cu \
-o ./tf_sampling_g.cu.o \
-c -O2 -DGOOGLE_CUDA=1 \
-x cu -Xcompiler \
-fPIC

g++ \
-std=c++11 \
-shared \
./tf_sampling.cpp \
./tf_sampling_g.cu.o \
-o ./tf_sampling_so.so \
-I $CUDA_ROOT/include \
-L $CUDA_ROOT/lib64/ \
-fPIC ${TF_CFLAGS} ${TF_LFLAGS} -O2
