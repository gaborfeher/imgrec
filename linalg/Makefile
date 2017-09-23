NVCC = /usr/local/cuda-8.0/bin/nvcc
CXX = clang++

# gtest setup
GTEST_DIR = ../googletest/downloaded_deps/googletest/googletest/
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h
CPPFLAGS += -isystem $(GTEST_DIR)/include
CXXFLAGS += -g -Wall -Wextra -pthread

.PHONY: get_gtest clean

build_gtest :
	$(MAKE) -C ../googletest downloaded_deps/googletest

../googletest/gtest_main.a :
	$(MAKE) -C ../googletest gtest_main.a


matrix.o: DeviceMatrix.cu HostMatrix.cpp DeviceMatrix.h HostMatrix.h
	$(NVCC) DeviceMatrix.cu HostMatrix.cpp \
		--lib \
		--output-file=matrix.o \
		-Wno-deprecated-gpu-targets \
		--compiler-bindir=$(CXX) \
		--std=c++11

hello: hello.cpp matrix.o
	$(CXX) hello.cpp matrix.o -ohello  --std=c++11 -L/usr/local/cuda-8.0/lib64 -lcudart

matrix_unittest.o: matrix_unittest.cc DeviceMatrix.h HostMatrix.h build_gtest
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c matrix_unittest.cc

matrix_unittest: matrix_unittest.o ../googletest/gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@

clean:
	rm -f ./hello ./matrix_unittest *.o
