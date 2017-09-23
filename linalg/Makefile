NVCC = /usr/local/cuda-8.0/bin/nvcc
CXX = clang++
BIN_DIR=../bin/linalg
TEST_BIN_DIR=../bin/googletest

.PHONY: build_gtest clean test hello matrix_unittest

# START gtest setup (TODO: factor  out)
GTEST_DIR = ../googletest/downloaded_deps/googletest/googletest/
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h
CPPFLAGS += -isystem $(GTEST_DIR)/include
CXXFLAGS += -g -Wall -Wextra -pthread --std=c++11

build_gtest :
	$(MAKE) -C ../googletest downloaded_deps/googletest

$(TEST_BIN_DIR)/gtest_main.a :
	$(MAKE) -C ../googletest $@

# END gtest setup

$(BIN_DIR) :
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/matrix.o: device_matrix.cu host_matrix.cc device_matrix.h host_matrix.h $(BIN_DIR)
	$(NVCC) device_matrix.cu host_matrix.cc \
		--lib \
		--output-file=$@ \
		-Wno-deprecated-gpu-targets \
		--compiler-bindir=$(CXX) \
		--std=c++11

$(BIN_DIR)/hello: hello.cc $(BIN_DIR)/matrix.o
	$(CXX) $^ -o $@  --std=c++11 -L/usr/local/cuda-8.0/lib64 -lcudart

$(BIN_DIR)/matrix_unittest.o: matrix_unittest.cc device_matrix.h host_matrix.h build_gtest $(BIN_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c matrix_unittest.cc -o $@

$(BIN_DIR)/matrix_unittest: $(BIN_DIR)/matrix_unittest.o $(BIN_DIR)/matrix.o $(TEST_BIN_DIR)/gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread -L/usr/local/cuda-8.0/lib64 -lcudart $^ -o $@

matrix_unittest: $(BIN_DIR)/matrix_unittest
	$(BIN_DIR)/matrix_unittest

hello: $(BIN_DIR)/hello

clean:
	rm -rf $(BIN_DIR)
