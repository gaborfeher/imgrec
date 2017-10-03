GTEST_DIR = googletest/downloaded_deps/googletest/googletest/
MAIN_GTEST_HEADER = $(GTEST_DIR)/include/gtest/gtest.h
NVCC = /usr/local/cuda/bin/nvcc
CUDA_LIB=/usr/local/cuda/lib64
CXX = clang++
CPPFLAGS += -isystem $(GTEST_DIR)/include -I .
CXXFLAGS += -g -Wall -Wextra -pthread --std=c++11 -L$(CUDA_LIB)

.PHONY: clean clean_all

clean:
	rm -Rf bin

clean_all: clean
	$(MAKE) -C googletest clean

# The header file is used as a marker for the whole gtest build.
$(MAIN_GTEST_HEADER) :
	$(MAKE) -C googletest downloaded_deps/googletest

bin/googletest/gtest_main.a : $(MAIN_GTEST_HEADER)
	mkdir -p bin/googletest
	$(MAKE) -C googletest ../$@


bin/linalg/matrix.o: linalg/device_matrix.cu linalg/device_matrix.h
	mkdir -p bin/linalg
	$(NVCC) $(filter %.cu %.cc,$^) \
		--include-path=. \
		--lib \
		--output-file=$@ \
		-Wno-deprecated-gpu-targets \
		--compiler-bindir=$(CXX) \
		--std=c++11

bin/linalg/hello: linalg/hello.cc bin/linalg/matrix.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lcudart $^ -o $@

bin/linalg/matrix_unittest.o: linalg/matrix_unittest.cc linalg/device_matrix.h $(MAIN_GTEST_HEADER)
	mkdir -p bin/linalg
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(filter %.cu %.cc %.o,$^) -o $@

bin/linalg/matrix_unittest: bin/linalg/matrix_unittest.o bin/linalg/matrix.o bin/googletest/gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread -lcudart $^ -o $@

matrix_unittest: bin/linalg/matrix_unittest
	$<

hello: bin/linalg/hello
	$<

bin/cnn/%.o: cnn/%.cc cnn/%.h
	mkdir -p bin/cnn
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(filter %.cu %.cc %.o,$^) -o $@

bin/cnn/learn_unittest.o: cnn/learn_unittest.cc $(MAIN_GTEST_HEADER)
	mkdir -p bin/cnn
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(filter %.cu %.cc %.o,$^) -o $@

bin/cnn/learn: cnn/learn.cc cnn/*.h bin/cnn/convolutional_layer.o bin/cnn/fully_connected_layer.o bin/cnn/model.o bin/cnn/error_layer.o bin/cnn/sigmoid_layer.o bin/cnn/layer_stack.o bin/cnn/layer.o bin/linalg/matrix.o
	mkdir -p bin/cnn
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(filter %.cu %.cc %.o,$^) \
		-o $@ \
		-L/usr/local/cuda-8.0/lib64 -lcudart

bin/cnn/learn_unittest: bin/cnn/learn_unittest.o bin/cnn/fully_connected_layer.o bin/cnn/model.o bin/cnn/error_layer.o bin/cnn/sigmoid_layer.o bin/cnn/layer_stack.o bin/cnn/layer.o bin/linalg/matrix.o bin/googletest/gtest_main.a
	mkdir -p bin/cnn
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread -lcudart  $(filter %.cu %.cc %.o %.a,$^) -o $@

learn_unittest: bin/cnn/learn_unittest
	$<

learn: bin/cnn/learn
	$<
