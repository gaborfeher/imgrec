GTEST_DIR = googletest/downloaded_deps/googletest/googletest/
MAIN_GTEST_HEADER = $(GTEST_DIR)/include/gtest/gtest.h

CXX = clang++
CPPFLAGS += -isystem $(GTEST_DIR)/include -I .
CXXFLAGS += -g -Wall -Wextra -pthread --std=c++11
CXXLINKFLAGS += -L$(CUDA_LIB) -lpthread -lcudart

NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS += --include-path=. -Wno-deprecated-gpu-targets --compiler-bindir=$(CXX) --std=c++11
CUDA_LIB=/usr/local/cuda/lib64

SOURCES=$(wildcard **/*.cc) $(wildcard **/*.cu)

# PHONY targets
#######

.PHONY: clean clean_all test matrix_test learn_test error_layer_test convolutional_layer_test

test: matrix_test learn_test error_layer_test convolutional_layer_test

clean:
	rm -Rf bin

clean_all: clean
	$(MAKE) -C googletest clean

matrix_test: bin/linalg/matrix_test
	$<

learn_test: bin/cnn/learn_test
	$<

error_layer_test: bin/cnn/error_layer_test
	$<

convolutional_layer_test: bin/cnn/convolutional_layer_test
	$<

# Make sure GoogleTest is downloaded before compiling test targets:
#######

$(MAIN_GTEST_HEADER) :
	$(MAKE) -C googletest downloaded_deps/googletest

TEST_SOURCES=$(filter _test.cc,$SOURCES)
$(subst .cc,.d,TEST_SOURCES): $(MAIN_GTEST_HEADER)
$(subst .cc,.o,TEST_SOURCES): $(MAIN_GTEST_HEADER)

# Build GoogleTest

bin/googletest/gtest_main.a : $(MAIN_GTEST_HEADER)
	mkdir -p bin/googletest
	$(MAKE) -C googletest ../$@

# Generate .d files (header dependency lists), and then include them in this Makefile
# to make header dependencies automatically discovered:
#######

bin/%.d: %.cc
	mkdir -p $(dir bin/$*)
	@set -e; rm -f $@; \
		$(CXX) -MM $(CPPFLAGS) $(CXXFLAGS) $< > $@.$$$$; \
		sed 's,\($(notdir $(basename $*))\)\.o[ :]*,$(dir bin/$*)\1.o $@ : ,g' < $@.$$$$ > $@; \
		rm -f $@.$$$$

bin/%.cu.d: %.cu
	mkdir -p $(dir bin/$*)
	@set -e; rm -f $@; \
		$(NVCC) -M $(NVCCFLAGS) $< > $@.$$$$; \
		sed 's,\($(notdir $(basename $*))\)\.o[ :]*,$(dir bin/$*)\1.cu.o $@ : ,g' < $@.$$$$ | \
		grep -v '^[[:space:]]*/' > $@; \
		echo >> $@; \
		rm -f $@.$$$$

include $(addprefix bin/,$(subst .cc,.d,$(subst .cu,.cu.d,$(SOURCES))))

# Invoke NVCC and C++ compilers to get .o files:
#######

bin/%.cu.o: %.cu
	mkdir -p $(dir bin/$*)
	$(NVCC) $(filter %.cu %.cc,$^) \
		$(NVCCFLAGS) \
		--lib \
		--output-file=$@

bin/%.o: %.cc
	mkdir -p $(dir bin/$*)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(filter %.cu %.cc %.o,$^) -o $@

# Fully linked end-product binary files:
#######

bin/linalg/matrix_test: bin/linalg/matrix_test.o \
		bin/linalg/device_matrix.cu.o \
		bin/linalg/matrix_test_util.o \
		bin/googletest/gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CXXLINKFLAGS) $^ -o $@

bin/cnn/learn_test: bin/cnn/learn_test.o \
		bin/cnn/data_set.o \
		bin/cnn/error_layer.o \
		bin/cnn/fully_connected_layer.o \
		bin/cnn/l2_error_layer.o \
		bin/cnn/layer.o \
		bin/cnn/layer_stack.o \
		bin/cnn/layer_test_base.o \
		bin/cnn/model.o \
		bin/cnn/nonlinearity_layer.o \
		bin/linalg/device_matrix.cu.o \
		bin/linalg/matrix_test_util.o \
		bin/googletest/gtest_main.a
	mkdir -p bin/cnn
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CXXLINKFLAGS) $^ -o $@

bin/cnn/error_layer_test: bin/cnn/error_layer_test.o \
		bin/cnn/error_layer.o \
		bin/cnn/l2_error_layer.o \
		bin/cnn/layer.o \
		bin/cnn/layer_test_base.o \
		bin/cnn/softmax_error_layer.o \
		bin/linalg/device_matrix.cu.o \
		bin/linalg/matrix_test_util.o \
		bin/googletest/gtest_main.a
	mkdir -p bin/cnn
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CXXLINKFLAGS) $^ -o $@

bin/cnn/convolutional_layer_test: bin/cnn/convolutional_layer_test.o \
		bin/cnn/convolutional_layer.o \
		bin/cnn/data_set.o \
		bin/cnn/error_layer.o \
		bin/cnn/fully_connected_layer.o \
		bin/cnn/l2_error_layer.o \
		bin/cnn/layer.o \
		bin/cnn/layer_stack.o \
		bin/cnn/layer_test_base.o \
		bin/cnn/model.o \
		bin/cnn/nonlinearity_layer.o \
		bin/cnn/reshape_layer.o \
		bin/cnn/softmax_error_layer.o \
		bin/linalg/device_matrix.cu.o \
		bin/linalg/matrix_test_util.o \
		bin/googletest/gtest_main.a
	mkdir -p bin/cnn
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CXXLINKFLAGS) $^ -o $@

