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

bin/linalg/matrix_test.o: linalg/matrix_test.cc linalg/device_matrix.h $(MAIN_GTEST_HEADER)
	mkdir -p bin/linalg
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(filter %.cu %.cc %.o,$^) -o $@

bin/linalg/matrix_test: bin/linalg/matrix_test.o bin/linalg/matrix.o bin/googletest/gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread -lcudart $^ -o $@

matrix_test: bin/linalg/matrix_test
	$<

hello: bin/linalg/hello
	$<

bin/cnn/%.o: cnn/%.cc cnn/%.h
	mkdir -p bin/cnn
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(filter %.cu %.cc %.o,$^) -o $@

bin/cnn/%_test.o: cnn/%_test.cc linalg/device_matrix.h $(MAIN_GTEST_HEADER)
	mkdir -p bin/cnn
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(filter %.cu %.cc %.o,$^) -o $@

bin/cnn/learn_test: bin/cnn/learn_test.o bin/cnn/fully_connected_layer.o bin/cnn/model.o bin/cnn/l2_error_layer.o bin/cnn/error_layer.o bin/cnn/nonlinearity_layer.o bin/cnn/layer_stack.o bin/cnn/layer.o bin/linalg/matrix.o bin/cnn/layer_test_base.o bin/cnn/reshape_layer.o bin/googletest/gtest_main.a
	mkdir -p bin/cnn
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread -lcudart  $(filter %.cu %.cc %.o %.a,$^) -o $@

bin/cnn/error_layer_test: bin/cnn/error_layer_test.o bin/cnn/l2_error_layer.o bin/cnn/softmax_error_layer.o bin/cnn/error_layer.o bin/cnn/layer.o bin/linalg/matrix.o bin/cnn/layer_test_base.o bin/googletest/gtest_main.a
	mkdir -p bin/cnn
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread -lcudart  $(filter %.cu %.cc %.o %.a,$^) -o $@

bin/cnn/convolutional_layer_test: bin/cnn/convolutional_layer_test.o bin/cnn/fully_connected_layer.o bin/cnn/model.o bin/cnn/softmax_error_layer.o bin/cnn/l2_error_layer.o bin/cnn/error_layer.o bin/cnn/nonlinearity_layer.o bin/cnn/layer_stack.o bin/cnn/layer.o bin/linalg/matrix.o bin/cnn/convolutional_layer.o bin/cnn/reshape_layer.o bin/cnn/layer_test_base.o bin/googletest/gtest_main.a
	mkdir -p bin/cnn
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread -lcudart  $(filter %.cu %.cc %.o %.a,$^) -o $@

learn_test: bin/cnn/learn_test
	$<

error_layer_test: bin/cnn/error_layer_test
	$<

convolutional_layer_test: bin/cnn/convolutional_layer_test
	$<

