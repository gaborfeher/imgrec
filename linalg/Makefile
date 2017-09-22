NVCC = /usr/local/cuda-8.0/bin/nvcc
CXX = clang++

# START GoogleTest setup
# TODO: Move to own file, separate flags from other code. Threading?

GTEST_DIR = ../deps/googletest-release-1.8.0/googletest/

# Flags passed to the preprocessor.
# Set Google Test's header directory as a system directory, such that
# the compiler doesn't generate warnings in Google Test headers.
CPPFLAGS += -isystem $(GTEST_DIR)/include

# Flags passed to the C++ compiler.
CXXFLAGS += -g -Wall -Wextra -pthread

# All tests produced by this Makefile.  Remember to add new tests you
# created to the list.
TESTS = sample1_unittest

# All Google Test headers.  Usually you shouldn't change this
# definition.
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

# Builds gtest.a and gtest_main.a.

# Usually you shouldn't tweak such internal variables, indicated by a
# trailing _.
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest-all.cc

gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest_main.cc

gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

# END GoogleTest setup

matrix.o: DeviceMatrix.cu HostMatrix.cpp DeviceMatrix.h HostMatrix.h
	$(NVCC) DeviceMatrix.cu HostMatrix.cpp \
		--lib \
		--output-file=matrix.o \
		-Wno-deprecated-gpu-targets \
		--compiler-bindir=$(CXX) \
		--std=c++11

hello: hello.cpp matrix.o
	$(CXX) hello.cpp matrix.o -ohello  --std=c++11 -L/usr/local/cuda-8.0/lib64 -lcudart

matrix_unittest.o: matrix_unittest.cc DeviceMatrix.h HostMatrix.h $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c matrix_unittest.cc

matrix_unittest: matrix_unittest.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@

clean:
	rm -f ./hello ./matrix_unittest
	rm -f $(TESTS) gtest.a gtest_main.a *.o
