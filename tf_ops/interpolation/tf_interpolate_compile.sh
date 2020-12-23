#/bin/bash
#
# TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
# TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
# TF1.2
# g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC \
#     -I ${TF_INC} \
#     -I ${TF_INC}/external/nsync/public \
#     -I /usr/local/cuda-10.1/include -lcudart -L /usr/local/cuda-10.1/lib64/ \
#     -L ${TF_LIB} -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
TF_CFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CUDA_ROOT=/usr/local/cuda

g++ \
-std=c++11 \
-shared \
./tf_interpolate.cpp -o \
./tf_interpolate_so.so  \
-I $CUDA_ROOT/include \
-lcudart \
-L $CUDA_ROOT/lib64/ \
-fPIC ${TF_CFLAGS} ${TF_LFLAGS} -O2
